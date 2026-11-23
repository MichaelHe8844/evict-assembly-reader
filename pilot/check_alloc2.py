# -*- coding: utf-8 -*-
"""离线核对：附件零价值 + λ=0.3 之后，纯分配层面的 ANR 曲线。"""
import json
import numpy as np
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample, comp_status
from anchor_auto import extract
from alloc import build_units, greedy_density, best_single, token_spans
from arm_c import fit_budget

KEYS = {tuple(x["key"]) for x in json.load(open("../results/xmodel_keys.json"))}
tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
ds = [d for d in build_dataset("../data/dev.json")
      if (d["id"], d["row"], d["col"]) in KEYS]
print("样本", len(ds))
print("预算 | 装箱 | 贪心(相关) | 贪心(均匀) | token/预算")
for frac in (0.03, 0.05, 0.10, 0.20):
    v = {"h": [], "r": [], "u": []}
    tk = []
    for d in ds:
        doc, spans = tokenize_sample(tok, d["segs"])
        B = max(8, int(frac * len(doc)))
        idx, diag = extract(d["segs"], d["question"], max_cells=20)
        h = fit_budget(idx, d["segs"], spans, B)
        outs = []
        for wm in ("relevance", "uniform"):
            units, base = build_units(d["segs"], diag, spans, wmode=wm, budget=B)
            sel, info = greedy_density(units, spans, B, base=base)
            s2, v2 = best_single(units, spans, B)
            if v2 > info["value"] + 1e-9:
                sel = s2
            outs.append((wm, token_spans(sel, spans)))
        tk.append((len(outs[0][1]), B))
        for name, toks in [("h", h)] + outs:
            key = {"h": "h", "relevance": "r", "uniform": "u"}[name]
            m = np.zeros(len(doc), dtype=bool)
            m[list(toks)] = True
            v[key].append(all(x[0] is None or x[0]
                              for x in (comp_status(m, spans, d["group"].get(c, []))
                                        for c in "VRCU")))
    print("%4d%% | %.3f | %.3f | %.3f | %.0f/%.0f" % (
        int(frac * 100), np.mean(v["h"]), np.mean(v["r"]), np.mean(v["u"]),
        np.mean([x[0] for x in tk]), np.mean([x[1] for x in tk])))
