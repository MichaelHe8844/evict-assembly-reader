# -*- coding: utf-8 -*-
"""留出集端到端：三条件 EM、配对检验、与调参段的对照。

用法：python analyze_holdout.py ../results/holdout_arith_rescored.json
"""
import json
import sys
from math import comb

import numpy as np

from analyze_slice import em_close


def mcn(a, b):
    x = int((a & ~b).sum())
    y = int((b & ~a).sum())
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)


def ci(v, n=None):
    v = np.asarray(v, dtype=float)
    n = n or len(v)
    se = np.sqrt(v.mean() * (1 - v.mean()) / max(1, n))
    return v.mean(), v.mean() - 1.96 * se, v.mean() + 1.96 * se


def main():
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    seq = {}
    for r in d["records"]:
        seq.setdefault(r["cond"], []).append(r)
    n = min(len(v) for v in seq.values())
    ok = {c: np.array([em_close(seq[c][i]["pred"], seq[c][i]["gold"]) for i in range(n)])
          for c in seq}

    print("留出集 n=%d" % n)
    for c in seq:
        m, lo, hi = ci(ok[c])
        print("  %-11s EM %.3f  95%%CI [%.3f, %.3f]  ANR %.3f" % (
            c, m, lo, hi, np.mean([seq[c][i].get("anr", float("nan")) for i in range(n)])))

    for a, b in (("auto_f@20", "snap@20"), ("full", "auto_f@20"), ("full", "snap@20")):
        if a in ok and b in ok:
            x, y, p = mcn(ok[a], ok[b])
            print("  %-10s vs %-10s %3d : %-3d p=%.4f" % (a, b, x, y, p))

    print("\n实际保留比例: %.3f" % np.mean(
        [seq["auto_f@20"][i]["budget_tokens"] / seq["auto_f@20"][i]["n_doc"] for i in range(n)]))


if __name__ == "__main__":
    main()
