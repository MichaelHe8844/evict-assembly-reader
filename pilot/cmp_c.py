# -*- coding: utf-8 -*-
"""扩样本后 C 与 Ind 的配对检验。"""
import json
import sys
from math import comb

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))
LBL = sys.argv[3] if len(sys.argv) > 3 else "B vs A"


def get(d):
    return {(r["id"], round(float(r["gold"]), 6)): r for r in d["records"] if r["cond"] == "auto_f@20"}


a, b = get(A), get(B)
ks = [k for k in b if k in a]
x = sum(1 for k in ks if b[k]["correct"] and not a[k]["correct"])
y = sum(1 for k in ks if a[k]["correct"] and not b[k]["correct"])
m = x + y
p = min(1.0, 2 * sum(comb(m, i) for i in range(min(x, y) + 1)) / 2 ** m) if m else 1.0
na = sum(a[k]["correct"] for k in ks) / len(ks)
nb = sum(b[k]["correct"] for k in ks) / len(ks)
print("%s：n=%d  %.3f → %.3f（%+.3f）  改善 %d / 恶化 %d  McNemar p=%.4f" % (
    LBL, len(ks), na, nb, nb - na, x, y, p))
