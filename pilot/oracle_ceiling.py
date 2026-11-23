# -*- coding: utf-8 -*-
"""天花板测试：把 gold 操作数所在的两格强制装入，其余照常填满预算。

回答一个问题：如果“选哪两格”这件事做到完美，现有编译式读者能拿到多少？
两种情况在同一进程、同一读者、同一 prompt 模板下各跑一遍：
  base   : 现有 auto_f 装配（抽取器选候选，按优先级装到 B）
  oracle : 先整对装入 gold 操作数所在的锚点组，再用 base 的候选填满 B

用法：
  python oracle_ceiling.py --adapter ../results/lora_rft --budgets 3,5,10,20 \
      --ids ../results/e2e_af20_base.json --out ../results/oracle_ceiling.json
"""
import argparse
import json
import re
import sys
import time

import torch

sys.path.insert(0, ".")
from arm_a import tokenize_sample          # noqa: E402
from anchor_auto import extract            # noqa: E402
from common import build_dataset           # noqa: E402
from lora_budget import keep_text, run_prog, close, clean  # noqa: E402

KIND_PRIO = {"cell": 0, "row_header": 1, "col_name": 2, "col_header": 2,
             "unit": 2, "text": 3, "sep": 9}
NUM = re.compile(r"\d[\d,]*\.?\d*")


def parse_num(t):
    m = NUM.search(t.replace("$", " ").replace("%", " "))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def gold_literals(prog):
    """程序里的数值字面量，去掉 const_5 这类常量记号。"""
    p = re.sub(r"const_\d+", " ", prog or "")
    out = []
    for m in NUM.finditer(p):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def _decode(tok, doc, kept):
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


def oracle_text(tok, rec, frac, golds, cells_rc=None):
    segs = rec["segs"]
    doc, spans = tokenize_sample(tok, segs)
    n = len(doc)
    if frac is None:
        return tok.decode(doc)
    B = max(8, int(frac * n))

    mc = max(1, int((0.85 * B) / 6))
    idx, _ = extract(segs, rec["question"], max_cells=mc)
    base = [i for i in idx if spans[i][0] >= 0]

    # 要强制装入的单元格：优先用给定的 (r,c)，否则按 gold 数值匹配
    cells = []
    if cells_rc:
        for (r0, c0) in cells_rc:
            for j, s in enumerate(segs):
                if s["kind"] == "cell" and s["row"] == r0 and s["col"] == c0:
                    cells.append(j)
                    break
    else:
        for j, s in enumerate(segs):
            if s["kind"] != "cell":
                continue
            v = parse_num(s["text"])
            if v is None:
                continue
            if any(abs(v - g) <= 1e-3 * max(1.0, abs(g)) for g in golds):
                cells.append(j)

    forced = set()
    for j in cells[:2]:
        r, c = segs[j]["row"], segs[j]["col"]
        for k, s in enumerate(segs):
            if s["kind"] == "cell" and s["row"] == r and s["col"] == c:
                forced.add(k)
            elif s["kind"] == "row_header" and s["row"] == r:
                forced.add(k)
            elif s["kind"] in ("col_name", "col_header", "unit") and s["col"] == c:
                forced.add(k)

    out = set()

    def take(i):
        a, b = spans[i]
        if a < 0 or b <= a:
            return
        if len(out | set(range(a, b))) <= B:
            out.update(range(a, b))

    for i in sorted(forced, key=lambda k: KIND_PRIO.get(segs[k]["kind"], 5)):
        take(i)
    for i in sorted(base, key=lambda k: KIND_PRIO.get(segs[k]["kind"], 5)):
        take(i)
    return _decode(tok, doc, sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="../results/lora_rft")
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--ids", default="../results/e2e_af20_base.json")
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--out", default="../results/oracle_ceiling.json")
    ap.add_argument("--force_json", default="")
    ap.add_argument("--cond", default="base,oracle")
    ap.add_argument("--n_new", type=int, default=40)
    args = ap.parse_args()

    raw = {r["id"]: r for r in json.load(open(args.data, encoding="utf-8"))}
    force = json.load(open(args.force_json, encoding="utf-8")) if args.force_json else {}
    keep_ids = [r["id"] for r in json.load(open(args.ids, encoding="utf-8"))["records"]]

    rows, seen = [], set()
    for d in build_dataset(args.data):
        if d["id"] not in set(keep_ids) or d["id"] in seen:
            continue
        seen.add(d["id"])
        try:
            ans = float(d["exe_ans"])
        except (TypeError, ValueError):
            continue
        prog = ((raw.get(d["id"], {}).get("qa") or {}).get("program") or "")
        rows.append(dict(id=d["id"], rec=d, q=d["question"], ans=ans,
                         golds=gold_literals(prog)))
    print("评测 %d 题" % len(rows), flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    result, details = {}, {}
    for b in args.budgets.split(","):
        frac = float(b) / 100.0
        det, stat = [], {}
        t0 = time.time()
        for cond in args.cond.split(","):
            ok = 0
            for r in rows:
                if cond == "base":
                    ctx = keep_text(tok, r["rec"], frac)
                elif cond == "force":
                    rc = force.get(r["id"])
                    ctx = (keep_text(tok, r["rec"], frac) if not rc else
                           oracle_text(tok, r["rec"], frac, r["golds"], cells_rc=rc))
                else:
                    ctx = oracle_text(tok, r["rec"], frac, r["golds"])
                prompt = "文：\n" + ctx + "\n\n问：" + r["q"].strip() + "\n式："
                pi = tok(prompt, return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=1500).to("cuda")
                with torch.no_grad():
                    g = model.generate(**pi, max_new_tokens=args.n_new, do_sample=False,
                                       pad_token_id=tok.pad_token_id,
                                       stop_strings=["\n", "答"], tokenizer=tok)
                out = tok.decode(g[0][pi["input_ids"].shape[1]:], skip_special_tokens=True)
                v = run_prog(out)
                good = bool(close(v, r["ans"]))
                ok += good
                det.append(dict(id=r["id"], cond=cond, ok=good, val=v,
                                gold=r["ans"], ctx_tokens=len(tok(ctx)["input_ids"])))
            stat[cond] = dict(n=len(rows), em=ok / len(rows))
        result[b] = stat
        details[b] = det
        parts = " ".join("%s %.3f" % (k, v["em"]) for k, v in stat.items())
        print("预算 %-3s  %s  (%.0fs)" % (b, parts, time.time() - t0), flush=True)

    json.dump(dict(summary=result, detail=details), open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print("\n=== 汇总 ===")
    for b, v in result.items():
        print("%-3s " % b + "  ".join("%s %.3f" % (k, x["em"])
                                      for k, x in v.items()))


if __name__ == "__main__":
    main()
