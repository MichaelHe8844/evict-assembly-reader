# -*- coding: utf-8 -*-
"""直接回答：操作数已完整保留时，模型还答错多少？

在真实算术（7B 8-bit，单步）已有的 auto_f@20 逐样本结果上，重建每个样本的
保留集与两个操作数的锚点组，联表得到 P(正确 | 操作数齐全) 与 P(正确 | 缺操作数)。
"""
import json
import re
import numpy as np
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample
from anchor_auto import extract, table_index
from alloc import token_spans
from arm_c import fit_budget

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def nums(text):
    out = set()
    for m in NUM.findall(text or ""):
        try:
            out.add(abs(round(float(m.replace(",", "")), 6)))
        except ValueError:
            pass
    return out


tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")  # 与 7B 同词表
raw = {e.get("id"): e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
ds = build_dataset("../data/dev.json")
by_key = {}
for d in ds:
    try:
        k = (d["id"], round(float(d["exe_ans"]), 6))
    except (TypeError, ValueError):
        continue
    by_key.setdefault(k, d)

res = json.load(open("../results/cand2_confirm_rescored.json"))
recs = [r for r in res["records"] if r["cond"] == "auto_f@20"]
print("记录", len(recs))

cov, ok = [], []
for r in recs:
    d = by_key.get((r["id"], round(float(r["gold"]), 6)))
    if d is None:
        continue
    doc, spans = tokenize_sample(tok, d["segs"])
    B = max(8, int(0.20 * len(doc)))
    idx, diag = extract(d["segs"], d["question"], max_cells=20)
    kept = set(fit_budget(idx, d["segs"], spans, B))
    cn, un, rh, cell, _, _ = table_index(d["segs"])
    prog = ((raw.get(d["id"], {}) or {}).get("qa") or {}).get("program", "") or ""
    want = nums(prog)
    op_cells = [(rr, cc) for (rr, cc), si in cell.items()
                if want and (nums(d["segs"][si]["text"]) & want)]
    if len(op_cells) < 2:
        continue
    got = 0
    for (rr, cc) in op_cells[:2]:
        g = [cell[(rr, cc)]] + ([rh[rr]] if rr in rh else []) + \
            ([cn[cc]] if cc in cn else []) + ([un[cc]] if cc in un else [])
        if set(token_spans(g, spans)) <= kept:
            got += 1
    cov.append(got == 2)
    ok.append(bool(r["correct"]))

cov = np.array(cov); ok = np.array(ok)
print("可用样本 %d" % len(cov))
for name, m in (("两操作数齐全", cov), ("缺至少一个", ~cov)):
    if m.sum():
        print("  %-8s n=%3d  EM=%.3f" % (name, m.sum(), ok[m].mean()))
print("  全体        n=%3d  EM=%.3f" % (len(ok), ok.mean()))
if (~ok).sum():
    print("  错误中“操作数齐全”占 %.1f%%（%d/%d）" % (
        100 * (cov & ~ok).sum() / (~ok).sum(), (cov & ~ok).sum(), (~ok).sum()))
