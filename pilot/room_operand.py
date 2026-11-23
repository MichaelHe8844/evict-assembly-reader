# -*- coding: utf-8 -*-
"""第一步：运算完整性证据集合选择，还有多大空间？

对每个单步样本拆三个量：
  ceiling(B)   = 若**知道**两个操作数在哪，覆盖它们所需的最小代价 ≤ B 的比例（形状上限）
  oracle(B)    = 同上，实际用十字覆盖两个操作数
  heuristic(B) = 不知道操作数、只用抽取器的候选 + 相关性权重，贪心选形（当前可达到）
  current      = 只有单元格形态（本文 5.11 的四种装配），实测约 0
空间 = ceiling - heuristic（选择器还能改进多少）、heuristic - current（方法能拿到多少）
"""
import json
import re
import numpy as np
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample, comp_status
from anchor_auto import extract, table_index
from alloc import token_spans
from arm_c import slice_anchor, fit_budget

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")


def nums(text):
    out = set()
    for m in NUM.findall(text or ""):
        try:
            out.add(abs(round(float(m.replace(",", "")), 6)))
        except ValueError:
            pass
    return out


tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
raw = {e.get("id"): e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
ds = []
for d in build_dataset("../data/dev.json"):
    prog = ((raw.get(d["id"], {}) or {}).get("qa") or {}).get("program", "") or ""
    if SINGLE.match(prog.strip()):
        ds.append(d)
    if len(ds) >= 120:
        break
print("单步样本", len(ds))

rows = {b: {"ceil": [], "heur": [], "cur": [], "cost": []} for b in (0.03, 0.05, 0.10, 0.20)}

for d in ds:
    doc, spans = tokenize_sample(tok, d["segs"])
    L = len(doc)
    idx, diag = extract(d["segs"], d["question"], max_cells=20)
    cn, un, rh, cell, _, _ = table_index(d["segs"])
    want = nums(((raw.get(d["id"], {}) or {}).get("qa") or {}).get("program", "") or "")
    op_cells = [(r, c) for (r, c), si in cell.items()
                if want and (nums(d["segs"][si]["text"]) & want)]
    if len(op_cells) < 2:
        continue
    op_cells = op_cells[:2]

    # 判据：该操作数的锚点组（格 + 行头 + 列名 + 单位）是否被完整包含
    op_cross = []
    for (r, c) in op_cells:
        g = [cell[(r, c)]] + ([rh[r]] if r in rh else []) + \
            ([cn[c]] if c in cn else []) + ([un[c]] if c in un else [])
        op_cross.append(set(token_spans(g, spans)))
    oracle_tok = op_cross[0] | op_cross[1]
    oracle_cost = len(oracle_tok) / L

    # 候选：多形状——格、行切片、列切片。同一格的不同形状是互斥的备选。
    cand = []
    rsc = diag.get("rscore", {}) or {}
    csc = diag.get("cscore", {}) or {}
    for (r, c) in [(i, j) for i in diag.get("rows", []) for j in diag.get("cols", [])]:
        if (r, c) not in cell:
            continue
        w = 1.0 + 0.3 * (rsc.get(r, 0.0) + csc.get(c, 0.0))
        for shape in ("row", "col", "cell"):
            if shape == "cell":
                segs_i = [cell[(r, c)]] + ([rh[r]] if r in rh else []) + \
                         ([cn[c]] if c in cn else []) + ([un[c]] if c in un else [])
            else:
                segs_i = slice_anchor(d["segs"], r, c, shape)
            t = set(token_spans(segs_i, spans))
            if t:
                cand.append((t, w))

    for B_frac in rows:
        B = max(8, int(B_frac * L))
        r = rows[B_frac]
        r["cost"].append(oracle_cost)
        r["ceil"].append(len(oracle_tok) <= B)
        # 贪心选形：价值/边际成本
        cov, val = set(), 0.0
        rest = list(range(len(cand)))
        while True:
            best, bk = None, None
            for k in rest:
                t, w = cand[k]
                new = t - cov
                if not new or len(cov) + len(new) > B:
                    continue
                key = w / len(new)
                if bk is None or key > bk:
                    best, bk = k, key
            if best is None:
                break
            cov |= cand[best][0]
            val += cand[best][1]
            rest.remove(best)
        cover = sum(1 for oc in op_cross if oc <= cov)
        r["heur"].append(cover == 2)
        # 当前：只剩单元格
        cur = fit_budget(idx, d["segs"], spans, B)
        r["cur"].append(set(cur) >= (op_cross[0] | op_cross[1]))

print("\n预算 | 形状上限 ceiling | 可选可达 heuristic | 当前(单元格) | 覆盖两操作数的代价(中位)")
for B_frac in (0.03, 0.05, 0.10, 0.20):
    r = rows[B_frac]
    print("%4d%% | %.3f | %.3f | %.3f | %.1f%%" % (
        int(B_frac * 100), np.mean(r["ceil"]), np.mean(r["heur"]),
        np.mean(r["cur"]), 100 * np.median(r["cost"])))
