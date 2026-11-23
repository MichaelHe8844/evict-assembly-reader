# -*- coding: utf-8 -*-
"""预算剂量反应：每个预算档下「锚点装配」vs「锚点装配 + 强制保留」，同批题目。

用法：python analyze_budget_sel.py
依赖 results/e2e_af{3,5,7,10,20}_{base,sel}.json
"""
import json
import math

import numpy as np

P = [3, 5, 7, 10, 20]


def load(path, cond):
    d = json.load(open(path, encoding="utf-8"))
    return {(r["id"], round(float(r["gold"]), 6)): r
            for r in d["records"] if r["cond"] == cond}


def mcn(a, b):
    x = sum(1 for k in a if a[k] and not b[k])
    y = sum(1 for k in a if b[k] and not a[k])
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    print("%-6s %5s %8s %8s %8s %10s %9s" %
          ("预算", "n", "基线EM", "强制EM", "Δ", "配对", "p"))
    for b in P:
        try:
            A = load("../results/e2e_af%d_base.json" % b, "auto_f@%d" % b)
            B = load("../results/e2e_af%d_sel.json" % b, "auto_f@%d" % b)
        except FileNotFoundError:
            print("%-6s （缺数据）" % ("%d%%" % b))
            continue
        ks = [k for k in B if k in A]
        a = {k: bool(A[k]["correct"]) for k in ks}
        bb = {k: bool(B[k]["correct"]) for k in ks}
        x, y, p = mcn(bb, a)
        print("%-6s %5d %8.3f %8.3f %+8.3f %10s %9.2g" %
              ("%d%%" % b, len(ks), np.mean(list(a.values())), np.mean(list(bb.values())),
               np.mean(list(bb.values())) - np.mean(list(a.values())),
               "+%d/-%d" % (x, y), p))
        print("       token %.0f → %.0f | ANR %.3f → %.3f"
              % (np.mean([A[k]["budget_tokens"] for k in ks]),
                 np.mean([B[k]["budget_tokens"] for k in ks]),
                 np.mean([A[k]["anr"] for k in ks]),
                 np.mean([B[k]["anr"] for k in ks])))


if __name__ == "__main__":
    main()
