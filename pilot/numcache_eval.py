# -*- coding: utf-8 -*-
"""NumCache（官方 number-preserving compressor）与我们同一读者、同一判分的 20% 档对照。

流程：我们的文档构造（common.build_document）→ NumCache 压缩到 20% token 预算
→ 同一 gold 读者（lora_compile）在压缩文本上作答 → 与 gold 比。

用法：python numcache_eval.py [--limit N]
"""
import argparse
import json
import os
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_probe", "NumCache", "numcache_init"))

from common import build_document                      # noqa: E402
from pot_prompt import FEWSHOT, INSTR                  # noqa: E402
from number_preserving_compressor import NumberPreservingCompressor  # noqa: E402

MODEL = "E:/KV_models/Qwen2.5-3B-Instruct"
ADAPTER = "../results/lora_compile"
DATA = "../data/fin_test.json"
KEYS = "../results/_keys216.json"
OUT = "../results/numcache_20.json"

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def doc_text(segs):
    out = []
    for s in segs:
        t = s["text"] if isinstance(s, dict) else (s[0] if isinstance(s, (tuple, list)) else str(s))
        if t:
            out.append(t)
    return " ".join(out)


OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}


def op2infix(expr):
    """把 FinQA 的 subtract(a, b) / divide(a, b) 转成中缀算式（支持嵌套）。"""
    e = expr.strip()
    m = re.fullmatch(r"(add|subtract|multiply|divide)\((.+)\)", e, flags=re.I)
    if not m:
        return e
    op = m.group(1).lower()
    inner = m.group(2)
    # 顶层逗号切分
    depth, cut = 0, -1
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cut = i
            break
    if cut < 0:
        return e
    a = op2infix(inner[:cut])
    b = op2infix(inner[cut + 1:])
    return "(%s %s %s)" % (a, OPS[op], b)


def parse_expr(text):
    """从生成文本里取一个可求值的算式。"""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    cand = None
    for l in lines:
        l = re.sub(r"^(expression|answer)\s*[:：]\s*", "", l, flags=re.I)
        l = l.strip("` ")
        if re.fullmatch(r"[\d\.\+\-\*/\(\)\s,%]+", l):
            cand = l.replace(",", "").replace("%", "")
    if cand is None:
        m = re.search(r"(add|subtract|multiply|divide)\([\d\.,\s\(\)a-z]+\)", text, flags=re.I)
        if m:
            return op2infix(m.group(0))
        m = re.findall(r"[-\d\.\+\*/\(\)\s]{3,}", text.replace(",", ""))
        if m:
            cand = max(m, key=len).strip()
    return cand


def safe_eval(expr):
    if not expr:
        return None
    if not re.fullmatch(r"[\d\.\+\-\*/\(\)\s]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adapter", default="", help="留空表示基座模型（跨方法公平口径）")
    ap.add_argument("--ratio", type=float, default=0.20,
                    help="预算比例；1.0 表示不压缩（对照）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    comp = NumberPreservingCompressor(tokenizer_name=MODEL)

    keys = json.load(open(KEYS, encoding="utf-8"))
    gold = {k[0]: float(k[1]) for k in keys}
    raw = {e["id"]: e for e in json.load(open(DATA, encoding="utf-8"))}

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    global ADAPTER
    ADAPTER = args.adapter
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="cuda", trust_remote_code=True)
    if ADAPTER:
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    recs = []
    ids = [i for i in gold if i in raw]
    if args.limit:
        ids = ids[: args.limit]
    print("待评", len(ids), "题", flush=True)

    for n, qid in enumerate(ids):
        ex = raw[qid]
        segs = build_document(ex)
        text = doc_text(segs)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        budget = max(8, int(args.ratio * n_tok))
        if args.ratio >= 1.0:
            ctx = text
        else:
            try:
                res = comp.compress(text, budget)
                ctx = res.get("compressed_text") or res.get("text") or ""
            except Exception as e:
                ctx = text
                print("压缩失败", qid, e, flush=True)
        q = ex["qa"].get("question", "")
        msg = [{"role": "system", "content": "You are a financial analyst."},
               {"role": "user", "content": FEWSHOT + "\nDocument:\n" + ctx +
                "\nQuestion: " + q + "\n" + INSTR}]
        prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        val = safe_eval(parse_expr(gen))
        g = gold[qid]
        ok = (val is not None and abs(val - g) <= 1e-3 * max(1.0, abs(g)))
        recs.append({"id": qid, "budget_tok": budget, "n_tok": n_tok,
                     "gen": gen[:120], "pred": val, "gold": g, "ok": ok,
                     "ctx_chars": len(ctx)})
        if (n + 1) % 10 == 0:
            em = sum(r["ok"] for r in recs) / len(recs)
            print("  %d/%d  EM=%.3f" % (n + 1, len(ids), em), flush=True)
            json.dump({"records": recs}, open(OUT + ".partial.json", "w", encoding="utf-8"),
                      ensure_ascii=False)

    em = sum(r["ok"] for r in recs) / max(1, len(recs))
    json.dump({"em": em, "n": len(recs), "cond": "numcache@%.2f" % args.ratio, "records": recs},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("NumCache@20  n=%d  EM=%.3f" % (len(recs), em))


if __name__ == "__main__":
    main()
