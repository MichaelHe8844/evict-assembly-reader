# -*- coding: utf-8 -*-
"""只测定位模块本身：不给模型看，直接算 gold 操作数是否落在候选里。"""
import json
import sys

sys.path.insert(0, ".")
from ground import score_cells, pick_pair  # noqa: E402
from operand_loc import gold_cells, close  # noqa: E402

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

rows = []
for k, f, c in pairs:
    entry = raw.get(k[0], {})
    cells = gold_cells(entry)
    if not cells or any(x is None for x in cells):
        continue
    ranked = score_cells(entry, entry["qa"]["question"], topk=6)
    if not ranked:
        continue
    gold = [x for x in cells if x]

    def cover(lst):
        return all(any(close(g["v"], it[1]["v"], 1e-3) for it in lst) for g in gold)

    top1 = ranked[0][1]
    rows.append(dict(
        id=k[0], n_gold=len(gold),
        top1_any=any(close(g["v"], top1["v"], 1e-3) for g in gold),
        k4=cover(ranked[:4]), k6=cover(ranked),
        pair=cover(pick_pair(ranked)),
        model_full_all=f["correct"], model_comp_all=c["correct"],
        loc_full=None, loc_comp=None,
    ))

n = len(rows)
def rate(key):
    return sum(1 for r in rows if r[key]) / n

print("样本 %d（单步、gold 可定位到表格）" % n)
print("top-1 命中任一 gold 操作数 : %.3f" % rate("top1_any"))
print("top-4 覆盖全部 gold 操作数 : %.3f" % rate("k4"))
print("top-6 覆盖全部 gold 操作数 : %.3f" % rate("k6"))
print("pick_pair 覆盖全部         : %.3f" % rate("pair"))

two = [r for r in rows if r["n_gold"] >= 2]
print("\n两个操作数的子集 n=%d：top-4 全覆盖 %.3f | pick_pair 全覆盖 %.3f" % (
    len(two), sum(r["k4"] for r in two) / len(two), sum(r["pair"] for r in two) / len(two)))

# 模块与模型对照：模型"一次只锁定一个"的比例
print("\n对照（同为 %d 题）：模块 top-4 全覆盖 %.3f，而模型在全上下文下两个操作数都用到的比例见 5.13" % (n, rate("k4")))
