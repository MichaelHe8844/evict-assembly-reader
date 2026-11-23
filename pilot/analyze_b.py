# -*- coding: utf-8 -*-
"""A / B 对照：按用户要求的四个问题出数。

1) "两边都只中一个"的题，有多少变为全命中？
2) 原本全命中的题，有多少被改坏？
3) 全命中是否转化为最终答案正确？
4) B 相对 A 是否有全样本净收益？
另报逐操作数命中率与整对全命中率。
"""
import json
import sys
from math import comb

sys.path.insert(0, ".")
from transformers import AutoTokenizer  # noqa: E402
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract  # noqa: E402
from arm_c import fit_budget  # noqa: E402
from ground import cells_from_segs, rank, expand_cids, parse_selection  # noqa: E402
from operand_loc import gold_cells, used_numbers, close  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "../results/g_base3b.json"
TREAT = sys.argv[2] if len(sys.argv) > 2 else "../results/g_mod3b.json"
COND = sys.argv[3] if len(sys.argv) > 3 else "auto_f@20"

raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
ds = {}
for d in build_dataset("../data/dev.json"):
    try:
        ds[(d["id"], round(float(d["exe_ans"]), 6))] = d
    except (TypeError, ValueError):
        continue

A = {(r["id"], round(float(r["gold"]), 6)): r
     for r in json.load(open(BASE, encoding="utf-8"))["records"] if r["cond"] == COND}
B = {(r["id"], round(float(r["gold"]), 6)): r
     for r in json.load(open(TREAT, encoding="utf-8"))["records"] if r["cond"] == COND}
keys = [k for k in B if k in A]

tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")


def state(rec, cells, ranked=None, is_treat=False):
    """(整对是否全命中, 命中比例)。处置组要先把 C# 展开成数值。"""
    txt = rec.get("text", "")
    if is_treat and ranked:
        _, expr = parse_selection(txt)
        txt = expand_cids(expr or txt, ranked)
    us = used_numbers({"text": txt})
    g = [c for c in cells if c]
    if not g:
        return None, None
    k = sum(1 for c in g if any(close(u, c["v"], 1e-3) for u in us))
    return (k == len(g)), k / len(g)


rows = []
for k in keys:
    d = ds.get(k)
    if not d:
        continue
    doc, spans = tokenize_sample(tok, d["segs"])
    mc = max(1, int((0.85 * int(0.20 * len(doc))) / 6))
    idx, _ = extract(d["segs"], d["question"], max_cells=mc)
    Bg = max(8, int(0.20 * len(doc)))
    keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], d["segs"], spans, Bg))
    ranked = rank(cells_from_segs(d["segs"], spans, keep), d["question"], topk=6)
    cells = gold_cells(raw[k[0]])
    if not cells:
        continue
    sa, ra = state(A[k], cells)
    sb, rb = state(B[k], cells, ranked, is_treat=True)
    if sa is None or sb is None:
        continue
    rows.append(dict(k=k, sa=sa, sb=sb, ra=ra, rb=rb,
                     ca=A[k]["correct"], cb=B[k]["correct"]))

n = len(rows)
print("条件 %s，配对 %d 条" % (COND, n))


def pct(x):
    return "%.3f" % (x / n)


def st(r):
    if r["ra"] == 1.0:
        return "全命中"
    return "部分" if r["ra"] > 0 else "全未命中"


def st2(r):
    if r["rb"] == 1.0:
        return "全命中"
    return "部分" if r["rb"] > 0 else "全未命中"


tab = {}
for r in rows:
    tab[(st(r), st2(r))] = tab.get((st(r), st2(r)), 0) + 1
print("A→B 状态迁移：", {("%s→%s" % k): v for k, v in sorted(tab.items())})

part = [r for r in rows if st(r) == "部分"]
part_to_all = [r for r in part if st2(r) == "全命中"]
print("\n1) A 里「只中一个」%d 题 → B 全命中 %d 题（%.0f%%）；其中最终答对 %d 题（%.0f%%）" % (
    len(part), len(part_to_all), 100 * len(part_to_all) / max(1, len(part)),
    sum(1 for r in part_to_all if r["cb"]), 100 * sum(1 for r in part_to_all if r["cb"]) / max(1, len(part_to_all))))

allA = [r for r in rows if st(r) == "全命中"]
broken = [r for r in allA if st2(r) != "全命中"]
print("2) A 里全命中 %d 题 → B 被改坏 %d 题（%.0f%%）；其中原本答对现在仍对 %d 题" % (
    len(allA), len(broken), 100 * len(broken) / max(1, len(allA)),
    sum(1 for r in broken if r["ca"] and r["cb"])))

allB = [r for r in rows if st2(r) == "全命中"]
print("3) B 全命中 %d 题：EM %.3f（A 同组 %.3f）" % (
    len(allB), sum(1 for r in allB if r["cb"]) / max(1, len(allB)),
    sum(1 for r in allB if r["ca"]) / max(1, len(allB))))

ea = sum(1 for r in rows if r["ca"])
eb = sum(1 for r in rows if r["cb"])
x = sum(1 for r in rows if r["cb"] and not r["ca"])
y = sum(1 for r in rows if r["ca"] and not r["cb"])
m = x + y
pv = min(1.0, 2 * sum(comb(m, i) for i in range(min(x, y) + 1)) / 2 ** m) if m else 1.0
print("4) EM：A %d/%d=%.3f → B %d/%d=%.3f；B 对 A 错 %d、A 对 B 错 %d，p=%.3f" % (
    ea, n, ea / n, eb, n, eb / n, x, y, pv))

ra = sum(r["ra"] for r in rows) / n
rb = sum(r["rb"] for r in rows) / n
pa = sum(1 for r in rows if st(r) == "全命中") / n
pb = sum(1 for r in rows if st2(r) == "全命中") / n
print("\n逐操作数命中率：A %.3f → B %.3f（+%.1f 个百分点）" % (ra, rb, 100 * (rb - ra)))
print("整对全命中率：  A %.3f → B %.3f（+%.1f 个百分点）" % (pa, pb, 100 * (pb - pa)))
