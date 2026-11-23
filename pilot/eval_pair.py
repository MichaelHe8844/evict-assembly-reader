# -*- coding: utf-8 -*-
"""模块级评估：候选质量 + 「一对」假设的命中率（不经过模型）。"""
import json
import sys

sys.path.insert(0, ".")
import re  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract  # noqa: E402
from arm_c import fit_budget  # noqa: E402
from ground import rank, cells_from_segs  # noqa: E402
from operand_loc import gold_cells, close  # noqa: E402


def best_pair(ranked):
    """两个操作数通常只在一个维度上不同（同行不同列 / 同列不同行）。"""
    best = None
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            a, b = ranked[i][1], ranked[j][1]
            both = (a["r"] != b["r"]) and (a["c"] != b["c"])
            s = ranked[i][0] + ranked[j][0] + (0.2 if both else 2.0)
            if best is None or s > best[0]:
                best = (s, a, b)
    return best


raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
seen, pairs = set(), []
for f in ("../results/cand2_confirm_rescored.json", "../results/arm_c_real_rescored.json"):
    d = json.load(open(f, encoding="utf-8"))
    by = {}
    for r in d["records"]:
        by.setdefault((r["id"], round(float(r["gold"]), 6)), {})[r["cond"]] = r
    for k, c in by.items():
        if k in seen or "full" not in c or "auto_f@20" not in c:
            continue
        seen.add(k)
        pairs.append((k, c["full"], c["auto_f@20"]))

ds = {}
for d in build_dataset("../data/dev.json"):
    try:
        ds[(d["id"], round(float(d["exe_ans"]), 6))] = d
    except (TypeError, ValueError):
        continue

tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
n = top6 = pair = pair_single = 0
for k, f, c in pairs:
    d = ds.get(k)
    if not d:
        continue
    doc, spans = tokenize_sample(tok, d["segs"])
    if len(doc) < 64 or len(doc) > 1200:
        continue
    mc = max(1, int((0.85 * int(0.20 * len(doc))) / 6))
    idx, _ = extract(d["segs"], d["question"], max_cells=mc)
    B = max(8, int(0.20 * len(doc)))
    keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], d["segs"], spans, B))
    rk = rank(cells_from_segs(d["segs"], spans, keep), d["question"], topk=6)
    gc = gold_cells(raw[k[0]])
    g = [x for x in gc if x] if gc else []
    if not rk or not g:
        continue
    n += 1
    if all(any(close(gg["v"], r[1]["v"], 1e-3) for r in rk) for gg in g):
        top6 += 1
    if len(g) >= 2:
        bp = best_pair(rk)
        if bp and all(any(close(gg["v"], y["v"], 1e-3) for y in (bp[1], bp[2])) for gg in g):
            pair += 1
        # 单格题的对照：只要 top-1 命中
    if len(g) == 1 and close(rk[0][1]["v"], g[0]["v"], 1e-3):
        pair_single += 1

n2 = sum(1 for k, f, c in pairs if len([x for x in (gold_cells(raw[k[0]]) or []) if x]) >= 2)
n1 = sum(1 for k, f, c in pairs if len([x for x in (gold_cells(raw[k[0]]) or []) if x]) == 1)
print("n=%d  top-6 全覆盖 %.3f" % (n, top6 / n))
print("两操作数题 n=%d：最优一对命中 %.3f" % (n2, pair / max(1, n2)))
print("单操作数题 n=%d：top-1 命中 %.3f" % (n1, pair_single / max(1, n1)))
