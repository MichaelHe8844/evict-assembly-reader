"""注意力类基线的预算扫描分析。

对比同一批 117 题上 SnapKV / H2O 在 20/40/60/80% 预算下的：
  EM、整组数值留存率（v_all，即 num 字段）、ANR。
目的是判断 20% 档 0.017 是「预算太紧」还是「实现有问题」——
判据：EM 是否随预算单调回升，且与该档的 v_all 同量级。
"""
import collections
import json
import math
import os
import sys

BASE = "results/arith_big.json"
SWEEP = "results/sweep_attn.json"


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    R = collections.defaultdict(dict)
    for r in d["records"]:
        R[r["id"]].setdefault(r["cond"], r)
    return R


def stat(recs):
    n = len(recs)
    em = sum(1 for r in recs if r["correct"]) / n
    numr = sum(1 for r in recs if r.get("num")) / n
    anr = sum(1 for r in recs if r.get("anr")) / n
    return em, numr, anr, n


def mcn(R, a, b):
    ids = [k for k in R if a in R[k] and b in R[k]]
    x = sum(1 for k in ids if R[k][b]["correct"] and not R[k][a]["correct"])
    y = sum(1 for k in ids if R[k][a]["correct"] and not R[k][b]["correct"])
    m = x + y
    if m == 0:
        return x, y, 1.0
    k = min(x, y)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return x, y, p


def main():
    R = load(BASE)
    if os.path.exists(SWEEP):
        for k, v in load(SWEEP).items():
            R[k].update(v)
    conds = [c for c in ("full", "snap@20", "snap@40", "snap@60", "snap@80",
                         "h2o@20", "h2o@40", "h2o@60", "auto_f@20")
             if all(c in R[k] for k in R)]
    print("覆盖 %d 题\n" % len(R))
    print("%-10s %6s %6s %6s %6s" % ("条件", "EM", "数值全留", "ANR", "n"))
    for c in conds:
        recs = [R[k][c] for k in R]
        em, numr, anr, n = stat(recs)
        print("%-10s %6.3f %8.3f %6.3f %6d" % (c, em, numr, anr, n))

    print("\n预算放宽时 EM 是否单调回升（判据）")
    for fam in ("snap", "h2o"):
        rows = [(c, stat([R[k][c] for k in R])[0])
                for c in conds if c.startswith(fam + "@")]
        if not rows:
            continue
        seq = " → ".join("%s%%: %.3f" % (c.split("@")[1], v) for c, v in rows)
        mono = all(rows[i][1] <= rows[i + 1][1] + 1e-9 for i in range(len(rows) - 1))
        print("  %-4s %s   单调=%s" % (fam.upper(), seq, mono))

    print("\n各档 vs 本文@20（配对）")
    for c in conds:
        if c.startswith(("snap@", "h2o@")) and "auto_f@20" in R[list(R)[0]]:
            x, y, p = mcn(R, c, "auto_f@20")
            print("  %-9s 本文 +%d/-%d  p=%.3g" % (c, x, y, p))

    print("\n各档相对全上下文（配对：该档赢/输）")
    for c in conds:
        if c != "full":
            x, y, p = mcn(R, "full", c)
            print("  %-9s +%d/-%d  p=%.3g" % (c, x, y, p))


if __name__ == "__main__":
    main()
