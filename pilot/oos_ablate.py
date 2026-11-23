# -*- coding: utf-8 -*-
"""OOS 的增益来源隔离（消融矩阵的 C2/C3/C4）。

C1 完整模块（选对 + 判式 + 精确求值）= 已有结果（0.552）。
本脚本补三个条件，全部在同一批 87 题、同一个 3B 上：

  arith  C2：给定 OOS 选中的两格与 OOS 判定的算式，让模型自己算（去掉"脚本求值"）
  expr   C3：给定 OOS 选中的两格（精确值），让模型自己写算式，脚本求值（去掉"算子头"）
  pool   C4：把整个候选池以精确值文本喂给模型，让它自己选两格并写算式，脚本求值
              （同时去掉"成对头""算子头""结构化通道"，只留"精确值文本"）

判定：
  C2 ≈ C1         → 精确求值不是来源，功劳在算子头
  C2 << C1        → 脚本求值是主要来源
  C3 ≈ C1         → 算子头可被模型替代
  C4 ≈ C1         → 整个模块可被"把精确值写成文本"替代，模块贡献只是通道
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FORMULAS = [
    ("a+b", lambda a, b: a + b),
    ("a-b", lambda a, b: a - b),
    ("b-a", lambda a, b: b - a),
    ("a*b", lambda a, b: a * b),
    ("a/b", lambda a, b: a / b if b else None),
    ("b/a", lambda a, b: b / a if a else None),
    ("(a-b)/b", lambda a, b: (a - b) / b if b else None),
    ("(b-a)/a", lambda a, b: (b - a) / a if a else None),
]
TOL = 1e-3


def close(x, y):
    return x is not None and abs(x - y) <= TOL * max(1.0, abs(y))


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else ("%.6g" % v)


def nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", s or "")
            if x.replace(",", "").strip(".") not in ("", "-")]


def safe_eval(text):
    t = (text or "").replace(",", "").replace("$", "").replace("％", "%")
    t = t.split("=")[-1]
    t = re.sub(r"[^0-9.+\-*/() ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    try:
        v = eval(t, {"__builtins__": {}}, {})
    except Exception:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def cell_line(c):
    unit = (" (%s)" % c["unit"].strip()) if c.get("unit") else ""
    return "%s | %s%s = %s" % (c.get("row", ""), c.get("col", ""), unit, c["raw"])


def build_prompt(mode, q, r, i, j, kop):
    a, b = r["cands"][i], r["cands"][j]
    if mode == "arith":
        f = FORMULAS[kop][0]
        return ("Let a = %s and b = %s. Compute exactly: %s. "
                "Reply with only the number, no words, no units." % (fmt(a["v"]), fmt(b["v"]), f))
    if mode == "expr":
        return ("Table cells (row | column = value):\n%s\n%s\n\nQuestion: %s\n\n"
                "Write one arithmetic expression over exactly these two values that answers the "
                "question. Use only these two numbers. Output only the expression."
                % (cell_line(a), cell_line(b), q))
    if mode == "pool":
        pool = "\n".join(cell_line(c) for c in r["cands"])
        return ("Table cells (row | column = value):\n%s\n\nQuestion: %s\n\n"
                "Write one arithmetic expression using at most two of these values that answers "
                "the question. Output only the expression, never the words." % (pool, q))
    raise ValueError(mode)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["arith", "expr", "pool"])
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--pred", default="../results/oos_pred.json")
    ap.add_argument("--pairs", default="../results/r1_pairs_dev.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pred = {x["key"][0]: x for x in json.load(open(args.pred, encoding="utf-8"))}
    rows = {tuple(r["key"]): r for r in json.load(open(args.pairs, encoding="utf-8"))}
    items = []
    for k, r in rows.items():
        qid = k[0]
        if qid not in pred:
            continue
        if not any(lab for _, _, lab in r["pairs"]):
            continue
        items.append((qid, r))
    if args.limit:
        items = items[:args.limit]
    print("评测 %d 题，模式 %s" % (len(items), args.mode), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
    if args.load_8bit:
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                  device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    ok = 0
    recs = []
    for n, (qid, r) in enumerate(items, 1):
        p = pred[qid]
        i, j = p["pair"]
        kop = p["op"]
        prompt = build_prompt(args.mode, r["q"], r, i, j, kop)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if args.mode == "arith":
            v = nums(gen)
            val = v[-1] if v else None
        else:
            val = safe_eval(gen)
        good = close(val, r["key"][1])
        ok += good
        recs.append(dict(qid=qid, gen=gen[:120], val=val, gold=r["key"][1], ok=bool(good)))
        if n % 10 == 0:
            print("  %d/%d 累计 %.3f" % (n, len(items), ok / n), flush=True)

    print("\n模式 %s：EM %.4f（%d/%d）" % (args.mode, ok / len(items), ok, len(items)))
    if args.out:
        json.dump(recs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
