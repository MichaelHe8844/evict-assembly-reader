# -*- coding: utf-8 -*-
"""模块效果：EM、定位命中（操作数级）、引用格与 gold 的关系。"""
import json
import sys

sys.path.insert(0, ".")
from transformers import AutoTokenizer  # noqa: E402
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract  # noqa: E402
from arm_c import fit_budget  # noqa: E402
from ground import cells_from_segs, rank, expand_cids, used_cids  # noqa: E402
from operand_loc import gold_cells, used_numbers, close  # noqa: E402

raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
ds = {}
for d in build_dataset("../data/dev.json"):
    try:
        ds[(d["id"], round(float(d["exe_ans"]), 6))] = d
    except (TypeError, ValueError):
        continue

base = json.load(open("../results/g_base3b.json", encoding="utf-8"))
mod = json.load(open("../results/g_mod3b.json", encoding="utf-8"))
COND = sys.argv[1] if len(sys.argv) > 1 else "auto_f@20"
bk = {(r["id"], round(float(r["gold"]), 6)): r for r in base["records"] if r["cond"] == COND}
mk = {(r["id"], round(float(r["gold"]), 6)): r for r in mod["records"] if r["cond"] == COND}
keys = [k for k in mk if k in bk]


def hit(rec, cells, expand=False, ranked=None):
    txt = expand_cids(rec.get("text", ""), ranked) if expand else rec.get("text", "")
    us = used_numbers({"text": txt})
    g = [c for c in cells if c]
    if not g:
        return None
    k = sum(1 for c in g if any(close(u, c["v"], 1e-3) for u in us))
    return "all" if k == len(g) else ("none" if k == 0 else "part")


tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
res = {"base": [], "mod": []}
rows = []
for k in keys:
    d = ds.get(k)
    if not d:
        continue
    doc, spans = tokenize_sample(tok, d["segs"])
    mc = max(1, int((0.85 * int(0.20 * len(doc))) / 6))
    idx, _ = extract(d["segs"], d["question"], max_cells=mc)
    B = max(8, int(0.20 * len(doc)))
    keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], d["segs"], spans, B))
    ranked = rank(cells_from_segs(d["segs"], spans, keep), d["question"], topk=6)
    cells = gold_cells(raw[k[0]])
    if not cells:
        continue
    hb = hit(bk[k], cells)
    hm = hit(mk[k], cells, expand=True, ranked=ranked)
    if hb is None or hm is None:
        continue
    res["base"].append(hb)
    res["mod"].append(hm)
    rows.append((k, bk[k]["correct"], mk[k]["correct"], hb, hm,
                 used_cids(mk[k].get("text", ""), ranked)))

n = len(rows)


def rate(src, want):
    return sum(1 for x in src if x == want) / n


print("条件 %s，配对样本 n=%d" % (COND, n))
for name, src in (("基线", res["base"]), ("模块", res["mod"])):
    print("  %s 定位：全命中 %.3f | 部分 %.3f | 全未命中 %.3f" % (
        name, rate(src, "all"), rate(src, "part"), rate(src, "none")))
b = sum(1 for r in rows if r[1]) / n
m = sum(1 for r in rows if r[2]) / n
print("  EM：基线 %.3f → 模块 %.3f（%+d 题）" % (
    b, m, sum(1 for r in rows if r[2] and not r[1]) - sum(1 for r in rows if r[1] and not r[2])))
fix = sum(1 for r in rows if r[3] != "all" and r[4] == "all")
print("  定位由「不全」变为「全命中」：%d 题；反向 %d 题" % (
    fix, sum(1 for r in rows if r[3] == "all" and r[4] != "all")))
print("  模型实际引用的候选数：%s" % (
    {k: sum(1 for r in rows if len(r[5]) == k) for k in (0, 1, 2, 3)}))
# 定位正确但算错
loc_ok = [r for r in rows if r[4] == "all"]
print("  定位全命中的 %d 题里，EM %.3f" % (len(loc_ok), sum(1 for r in loc_ok if r[2]) / max(1, len(loc_ok))))
if COND != "full":
    keep = [r for r in rows if bk[r[0]]["num"]]
    print("  gold 值保留的 %d 题：定位全命中 基线 %.3f → 模块 %.3f" % (
        len(keep), sum(1 for r in keep if r[3] == "all") / max(1, len(keep)),
        sum(1 for r in keep if r[4] == "all") / max(1, len(keep))))
