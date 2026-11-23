# -*- coding: utf-8 -*-
"""多形状下的上限：不只用十字，也允许行切片/列切片/单元格。

对每个样本、每种形状、每个候选格，检查它是否**同时覆盖两个操作数**以及代价，
得到「该预算下存在可行形状」的比例 —— 这是任何选择器能达到的上限。
"""
import json
import re
import numpy as np
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample
from anchor_auto import extract, table_index
from alloc import token_spans
from arm_c import slice_anchor

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

budgets = (0.03, 0.05, 0.10, 0.20)
best = {s: {b: [] for b in budgets} for s in ("cell", "row", "col", "cross", "any")}
costs = {s: [] for s in ("row", "col", "cross")}

for d in ds:
    doc, spans = tokenize_sample(tok, d["segs"])
    L = len(doc)
    _, diag = extract(d["segs"], d["question"], max_cells=20)
    cn, un, rh, cell, _, _ = table_index(d["segs"])
    want = nums(((raw.get(d["id"], {}) or {}).get("qa") or {}).get("program", "") or "")
    op_cells = [(r, c) for (r, c), si in cell.items()
                if want and (nums(d["segs"][si]["text"]) & want)]
    if len(op_cells) < 2:
        continue
    op_cells = op_cells[:2]
    # 「某个操作数可复算」的判据是它的锚点组（格 + 行头 + 列名 + 单位）被完整包含，
    # 不是它自己的十字被包含——十字只对生成它的那个格子成立。
    op = []
    for (r, c) in op_cells:
        g = {cell[(r, c)]}
        if r in rh:
            g.add(rh[r])
        if c in cn:
            g.add(cn[c])
        if c in un:
            g.add(un[c])
        op.append(set(token_spans(g, spans)))

    # 每种形状：在抽取器给的候选格上生成，取“能覆盖两个操作数且代价最小”的那一个
    per_shape_min = {}
    for shape in ("cell", "row", "col", "cross"):
        mn = None
        for (r, c) in [(i, j) for i in diag.get("rows", []) for j in diag.get("cols", [])]:
            if (r, c) not in cell:
                continue
            if shape == "cell":
                s = list(op[0] | op[1]) if False else [cell[(r, c)]] + \
                    ([rh[r]] if r in rh else []) + ([cn[c]] if c in cn else []) + \
                    ([un[c]] if c in un else [])
            else:
                s = slice_anchor(d["segs"], r, c, shape)
            t = set(token_spans(s, spans))
            if not t:
                continue
            if all(o <= t for o in op):
                c_frac = len(t) / L
                mn = c_frac if mn is None else min(mn, c_frac)
        per_shape_min[shape] = mn
        if mn is not None:
            costs[shape].append(mn)
    for b in budgets:
        B = b
        for shape, mn in per_shape_min.items():
            best[shape][b].append(mn is not None and mn <= B)
        ok = any((mn is not None and mn <= B) for mn in per_shape_min.values())
        best["any"][b].append(ok)

print("覆盖两个操作数所需代价（中位/25 分位，占文档比例）")
for s in ("row", "col", "cross"):
    v = np.array(costs[s])
    if len(v) == 0:
        print("  %-6s n=0（该形状从未覆盖两个操作数）" % s)
        continue
    print("  %-6s n=%3d  中位 %.1f%%  25%% %.1f%%  最小 %.1f%%" % (
        s, len(v), 100 * np.median(v), 100 * np.percentile(v, 25), 100 * np.min(v)))

print("\n预算 | 仅单元格 | 仅行切片 | 仅列切片 | 仅十字 | 多形状(取最优)")
for b in budgets:
    print("%4d%% | %.3f | %.3f | %.3f | %.3f | %.3f" % (
        int(b * 100), np.mean(best["cell"][b]), np.mean(best["row"][b]),
        np.mean(best["col"][b]), np.mean(best["cross"][b]), np.mean(best["any"][b])))
