# -*- coding: utf-8 -*-
"""精确最优值校验：用 MILP 求 AGBA 的真最优，量出贪心的真实次优度。

为什么不用独占代价松弛：单元共享行头，独占代价远小于真实边际代价，
松弛在紧预算下极松（3% 档差约 50%），只能当弱上界。候选数中位 6、最大 20，
直接求整数最优完全可行（HiGHS）。
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample
from anchor_auto import extract
from alloc import build_units, greedy_density, best_single, token_spans

KEYS = {tuple(x["key"]) for x in json.load(open("../results/xmodel_keys.json"))}
tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
ds = [d for d in build_dataset("../data/dev.json")
      if (d["id"], d["row"], d["col"]) in KEYS]


def solve_exact(units, spans, B, base=()):
    toks = sorted({t for u in units for t in token_spans(u["segs"], spans)})
    idx_of = {t: k for k, t in enumerate(toks)}
    n, m = len(units), len(toks)
    if n == 0:
        return 0.0
    nv = n + m
    c = np.zeros(nv)
    c[:n] = [-u["w"] for u in units]
    rows, lo, hi = [], [], []
    # x_i - y_t <= 0
    for i, u in enumerate(units):
        for t in token_spans(u["segs"], spans):
            r = np.zeros(nv)
            r[i] = 1.0
            r[n + idx_of[t]] = -1.0
            rows.append(r)
            lo.append(-np.inf)
            hi.append(0.0)
    # sum y_t <= B - |base|
    r = np.zeros(nv)
    r[n:] = 1.0
    rows.append(r)
    base_tok = len(set(token_spans(base, spans)))
    lo.append(-np.inf)
    hi.append(max(0, B - base_tok))
    A = np.vstack(rows)
    res = milp(c=c, constraints=LinearConstraint(A, lo, hi),
               integrality=np.ones(nv), bounds=Bounds(0, 1),
               options=dict(time_limit=10, presolve=True))
    if res.status != 0:
        return float("nan")
    return float(-res.fun)


print("预算 | 贪心值 | 精确最优 | 平均差距 | 次优样本比例 | 最大差距")
for frac in (0.03, 0.05, 0.10, 0.20):
    gs, ops, gaps, bad, mx = [], [], [], 0, 0.0
    for d in ds:
        doc, spans = tokenize_sample(tok, d["segs"])
        B = max(8, int(frac * len(doc)))
        _, diag = extract(d["segs"], d["question"], max_cells=20)
        units, base = build_units(d["segs"], diag, spans, wmode="relevance", budget=B)
        sel, info = greedy_density(units, spans, B, base=base)
        s2, v2 = best_single(units, spans, B)
        gval = max(info["value"], v2)
        opt = solve_exact(units, spans, B, base)
        if opt != opt or opt <= 1e-9:
            continue
        gs.append(gval)
        ops.append(opt)
        gap = 1 - gval / opt
        gaps.append(gap)
        if gap > 1e-6:
            bad += 1
        mx = max(mx, gap)
    print("%4d%% | %.2f | %.2f | %.1f%% | %.1f%% | %.1f%%" % (
        int(frac * 100), np.mean(gs), np.mean(ops), 100 * np.mean(gaps),
        100 * bad / len(gs), 100 * mx))
