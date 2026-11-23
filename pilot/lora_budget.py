# -*- coding: utf-8 -*-
"""预算-准确率：把训练好的编译式读者放到压缩证据上评测。

压缩沿用论文的 auto_f 配方（与 r1_data.py 建池时完全一致）：
  结构抽取 -> 锚点 -> fit_budget 装到 B 个 token

用法：
  python lora_budget.py --adapter ../results/lora_full2 --n 126
"""
import argparse
import json
import math
import re
import sys
import time

import torch

sys.path.insert(0, ".")
from arm_a import tokenize_sample          # noqa: E402
from arm_c import fit_budget               # noqa: E402
from anchor_auto import extract            # noqa: E402
from common import build_dataset           # noqa: E402

OPS = {
    "add": lambda *a: sum(a),
    "subtract": lambda a, b: a - b,
    "multiply": lambda *a: math.prod(a),
    "divide": lambda a, b: a / b if b else None,
    "greater": lambda a, b: 1.0 if a > b else 0.0,
    "average": lambda *a: sum(a) / len(a),
    "exp": lambda a, b: a ** b,
}
SAFE = re.compile(r"^[0-9eE\.\,\-\+\(\)\sa-z_]+$")
CONST = re.compile(r"const_(\d+)")


def clean(prog):
    p = prog.strip().split("\n")[0].strip()
    p = re.split(r"答|式|问", p)[0].strip()
    p = p.split("=")[0].strip()
    return CONST.sub(r"\1", p)


def _split_steps(p):
    """按顶层逗号切步；括号内的逗号不切（参数分隔、千分位都在括号里）。"""
    out, depth, cur = [], 0, []
    for ch in p:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _eval_one(p):
    if not p or not SAFE.match(p):
        return None
    if "(" not in p:
        if not re.match(r"^[0-9eE\.\,\-\+\*/\s]+$", p):
            return None
        try:
            return float(eval(p, {"__builtins__": {}}))
        except Exception:
            return None
    try:
        return float(eval(p, {"__builtins__": {}}, dict(OPS)))
    except Exception:
        return None


def run_prog(prog):
    """单步程序直接求值；多步（逗号分隔 + #k 引用）按序求值，返回最后一步。"""
    p = clean(prog).replace("%", "")
    steps = _split_steps(p)
    if len(steps) > 1 or "#" in p:
        vals = []
        for s in steps:
            def sub(m):
                k = int(m.group(1))
                return repr(vals[k]) if k < len(vals) else "nan"
            s2 = re.sub(r"#(\d+)", sub, s)
            v = _eval_one(s2)
            if v is None:
                return None
            vals.append(v)
        return vals[-1] if vals else None
    return _eval_one(p)


def close(a, b, tol=1e-3):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def keep_text(tok, rec, frac):
    """按 auto_f 配方返回保留文本。frac=None 表示全文。"""
    segs = rec["segs"]
    doc, spans = tokenize_sample(tok, segs)
    n_doc = len(doc)
    if frac is None:
        return tok.decode(doc)
    B = max(8, int(frac * n_doc))
    mc = max(1, int((0.85 * B) / 6))
    idx, _ = extract(segs, rec["question"], max_cells=mc)
    keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], segs, spans, B))
    kept = [i for i in range(n_doc) if i in keep]
    chunks, cur = [], []
    for i in kept:
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                chunks.append(cur)
            cur = [i]
    if cur:
        chunks.append(cur)
    return " ".join(tok.decode([doc[i] for i in c]) for c in chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_full2")
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--ids", default="../results/e2e_af20_base.json")
    ap.add_argument("--n", type=int, default=126)
    ap.add_argument("--budgets", default="3,5,10,20,full")
    ap.add_argument("--out", default="../results/lora_budget.json")
    ap.add_argument("--budget_tag", action="store_true")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    keep_ids = {r["id"] for r in json.load(open(args.ids, encoding="utf-8"))["records"]}
    rows, seen = [], set()
    for d in build_dataset(args.data):
        if d["id"] not in keep_ids or d["id"] in seen:
            continue
        seen.add(d["id"])
        try:
            ans = float(d["exe_ans"])
        except (TypeError, ValueError):
            continue
        rows.append(dict(id=d["id"], rec=d, q=d["question"], ans=ans))
        if len(rows) >= args.n:
            break
    print("评测 %d 题" % len(rows), flush=True)

    print("载入 4-bit 底座 + 适配器", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    result, details = {}, {}
    for b in args.budgets.split(","):
        frac = None if b == "full" else float(b) / 100.0
        t0 = time.time()
        ok_p = ok_e = 0
        det = []
        for r in rows:
            ctx = keep_text(tok, r["rec"], frac)
            tag = ("预算：%s%%\n" % b) if args.budget_tag else ""
            prompt = tag + "文：\n" + ctx + "\n\n问：" + r["q"].strip() + "\n式："
            pi = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     truncation=True, max_length=1500).to("cuda")
            with torch.no_grad():
                g = model.generate(**pi, max_new_tokens=40, do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   stop_strings=["\n", "答"], tokenizer=tok)
            out = tok.decode(g[0][pi["input_ids"].shape[1]:], skip_special_tokens=True)
            v = run_prog(out)
            ok_p += v is not None
            ok_e += close(v, r["ans"])
            det.append(dict(id=r["id"], gen=clean(out), val=v, gold=r["ans"],
                            ok=bool(close(v, r["ans"])), ctx_tokens=len(tok(ctx)["input_ids"])))
        n = len(rows)
        result[b] = dict(n=n, exec=ok_p / n, em=ok_e / n,
                         med_ctx=int(sorted(d["ctx_tokens"] for d in det)[n // 2]),
                         sec=round(time.time() - t0, 1))
        print("预算 %-4s 证据中位 %4d tok  可执行 %.3f  EM %.3f  (%.0fs)"
              % (b, result[b]["med_ctx"], result[b]["exec"], result[b]["em"],
                 result[b]["sec"]), flush=True)
        details[b] = det

    json.dump(dict(summary=result, detail=details), open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print("\n=== 预算-准确率 ===")
    for b, v in result.items():
        print("%-4s 中位证据 %4d  EM %.3f" % (b, v["med_ctx"], v["em"]))


if __name__ == "__main__":
    main()
