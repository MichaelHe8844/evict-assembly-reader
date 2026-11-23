# -*- coding: utf-8 -*-
"""小测试：3B 在 few-shot PoT 下能否处理 FinQA 原始多步问题。

对照 zero-shot PoT。指标：合法算式率、EM。
"""
import re
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample
from model_utils import load_model

FEWSHOT = """Example 1
Document:
item | 2019 | 2020
revenue | 120 | 150
Question: what is the percent increase in revenue from 2019 to 2020?
Expression: (150 - 120) / 120 * 100

Example 2
Document:
segment | amount
north america | 400
europe | 250
Question: what is the total amount for all segments?
Expression: 400 + 250

Example 3
Document:
metric | 2021
operating income | 350
net revenue | 2450
Question: what is operating income as a percentage of net revenue?
Expression: 350 / 2450 * 100
"""

INSTR = ("Write ONE Python arithmetic expression that computes the answer, using only "
         "numbers that appear in the document. Output the expression only.")

EXPR_CHARS = re.compile(r"[-+*/()0-9.\s]{3,200}")


def safe_eval(text):
    m = EXPR_CHARS.search(text or "")
    if not m:
        return None
    expr = m.group(0).strip()
    if not re.search(r"\d", expr):
        return None
    try:
        v = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def exact_em(pred, gold):
    if pred is None:
        return False
    return abs(pred - gold) <= max(0.01, 0.005 * abs(gold))


def em(pred, gold):
    """允许百分比/比率两种约定（×100 / ÷100）。"""
    if pred is None:
        return False
    return exact_em(pred, gold) or exact_em(pred / 100.0, gold) or exact_em(pred * 100.0, gold)


def scale_only(pred, gold):
    """差分只来自 100 倍量纲约定，而非取数错误。"""
    return (not exact_em(pred, gold)) and em(pred, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--n_new", type=int, default=40)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--ctx", default="full", choices=["full", "evidence"],
                    help="full=整页(行式表+全部正文)；evidence=FinQA 官方证据片段")
    ap.add_argument("--single_step", action="store_true", help="只取单步算术样本")
    ap.add_argument("--raw_json", default="../data/dev.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    m = load_model(args.model, args.load_4bit, quant8=args.load_8bit).eval()

    ds = build_dataset("../data/dev.json")
    raw = {e.get("id"): e for e in json.load(open(args.raw_json, encoding="utf-8"))}
    single = re.compile(r"^[a-z]+\s*\([^()]*\)$")
    rows = []
    for d in ds:
        if len(rows) >= args.limit:
            break
        try:
            gold = float(d["exe_ans"])
        except (TypeError, ValueError):
            continue
        if args.single_step:
            prog = (raw.get(d["id"], {}).get("qa", {}) or {}).get("program", "") or ""
            if not single.match(prog.strip()):
                continue
        ids, _ = tokenize_sample(tok, d["segs"])
        if len(ids) > 1400:
            continue
        if args.ctx == "evidence":
            mi = raw.get(d["id"], {}).get("qa", {}).get("model_input", []) or []
            doc = "\n".join(t for _, t in mi)
        else:
            doc = tok.decode(ids)
        if args.fewshot:
            body = FEWSHOT + "\nDocument:\n" + doc
        else:
            body = "Document:\n" + doc
        msg = ("<|im_start|>system\nYou are a financial analyst.<|im_end|>\n<|im_start|>user\n"
               + body + "\n\nQuestion: " + d["question"] + "\n" + INSTR +
               "<|im_end|>\n<|im_start|>assistant\n")
        x = tok(msg, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            o = m.generate(x, max_new_tokens=args.n_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, x.shape[1]:], skip_special_tokens=True).strip()
        pred = safe_eval(txt)
        rows.append(dict(q=d["question"][:60], gold=gold, out=txt[:60],
                         pred=pred, ok=em(pred, gold), exact=exact_em(pred, gold),
                         sc=scale_only(pred, gold)))
        torch.cuda.empty_cache()

    n = len(rows)
    parse = np.mean([r["pred"] is not None for r in rows])
    acc = np.mean([r["ok"] for r in rows])
    acc_e = np.mean([r["exact"] for r in rows])
    sc = np.mean([r["sc"] for r in rows])
    print("fewshot=%s  n=%d  合法算式率 %.3f  EM(严格) %.3f  EM(尺度归一) %.3f  "
          "其中仅差量纲 %.3f" % (args.fewshot, n, parse, acc_e, acc, sc))
    print("\n前 10 例：")
    for r in rows[:10]:
        mark = "OK " if r["ok"] else "   "
        print("%s gold=%-10s pred=%-10s | %s" % (
            mark, r["gold"], r["pred"], r["out"].replace("\n", " ")))


if __name__ == "__main__":
    main()
