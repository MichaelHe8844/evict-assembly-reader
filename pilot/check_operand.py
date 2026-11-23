# -*- coding: utf-8 -*-
"""算术任务的分配检验（结构层，不需要生成）。

审稿人担心的是：贪心优先装高分单元，可能漏掉第二个操作数。这个担心可以在
结构层直接验证——用 gold 程序里的数值实参定位操作数单元格，统计「两个操作数
是否都被保留」。与模型无关，比在 7B 上跑几十条噪声样本更有信息量。
"""
import json
import re
import numpy as np
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample
from anchor_auto import extract, table_index
from alloc import build_units, greedy_density, greedy_static, rank_only, best_single, token_spans
from arm_c import fit_budget, slice_anchor

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


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
SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")
ds = []
for d in build_dataset("../data/dev.json"):
    prog = ((raw.get(d["id"], {}) or {}).get("qa") or {}).get("program", "") or ""
    if SINGLE.match(prog.strip()):
        ds.append(d)
    if len(ds) >= 120:
        break
print("单步样本", len(ds))

print("预算 | 装箱 | 排序 | 静态密度 | 动态密度")
for frac in (0.03, 0.05, 0.10, 0.20):
    res = {k: [] for k in ("pack", "rank", "stat", "knap")}
    for d in ds:
        doc, spans = tokenize_sample(tok, d["segs"])
        B = max(8, int(frac * len(doc)))
        off = 0
        idx, diag = extract(d["segs"], d["question"], max_cells=20)
        # 操作数单元格：程序里的数值实参 & 落在表里的
        cn, un, rh, cell, _, _ = table_index(d["segs"])
        want = nums(((raw.get(d["id"], {}) or {}).get("qa") or {}).get("program", "") or "")
        op_cells = [(r, c) for (r, c), si in cell.items()
                    if want and (nums(d["segs"][si]["text"]) & want)]
        if len(op_cells) < 2:
            continue
        # 每个操作数取一个十字，作为“该操作数可复算”的判据
        need = set()
        for (r, c) in op_cells[:2]:
            need |= set(slice_anchor(d["segs"], r, c, "both"))

        u2, b2 = build_units(d["segs"], diag, spans, "relevance", budget=B)
        toks = {
            "pack": fit_budget(idx, d["segs"], spans, B),
            "rank": token_spans(rank_only(u2, spans, B, base=b2)[0], spans),
            "stat": token_spans(greedy_static(u2, spans, B, base=b2)[0], spans),
        }
        s3, info = greedy_density(u2, spans, B, base=b2)
        s4, v4 = best_single(u2, spans, B)
        if v4 > info["value"] + 1e-9:
            s3 = s4
        toks["knap"] = token_spans(s3, spans)

        for k, t in toks.items():
            kept = set(t)
            ok = all(all(tt in kept for tt in range(*spans[i])) for i in need if spans[i][0] >= 0)
            res[k].append(ok)
    print("%4d%% | %.3f | %.3f | %.3f | %.3f  (n=%d)" % (
        int(frac * 100),
        np.mean(res["pack"]), np.mean(res["rank"]),
        np.mean(res["stat"]), np.mean(res["knap"]), len(res["pack"])))
