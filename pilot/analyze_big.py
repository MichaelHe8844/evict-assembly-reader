# -*- coding: utf-8 -*-
"""放大后的主结果（单步全集）汇总：EM / Wilson CI / ANR / McNemar。

口径与论文一致：每条 gold 单元格一条记录，取每题**首条**记录做配对比较。
用法：python analyze_big.py ../results/arith_big.json
"""
import json
import math
import sys

CONDS = ["full", "snap@20", "h2o@20", "auto_f@20", "auto_n@20",
         "numb@20", "arow@20", "acol@20"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcn(a, b):
    x = sum(1 for k in a if b[k] and not a[k])
    y = sum(1 for k in a if a[k] and not b[k])
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else "../results/arith_big.json"
    d = json.load(open(p, encoding="utf-8"))
    recs = d["records"]
    # 每题首条（id 相同、gold 可能不同，按出现顺序取第一个）
    first = {}
    for r in recs:
        first.setdefault(r["id"], {}).setdefault(r["cond"], r)
    ids = [i for i in first if all(c in first[i] for c in CONDS)]
    print("记录 %d，唯一问题 %d（四条件齐全 %d）" % (len(recs), len(first), len(ids)))
    print("%-11s %5s %6s %-16s %6s" % ("条件", "n", "EM", "95% CI", "ANR"))
    corr = {}
    for c in CONDS:
        ok = sum(1 for i in ids if first[i][c]["correct"])
        anr = sum(1 for i in ids if first[i][c]["anr"]) / max(1, len(ids))
        lo, hi = wilson(ok, len(ids))
        print("%-11s %5d %6.3f [%.2f, %.2f]      %6.3f"
              % (c, len(ids), ok / len(ids), lo, hi, anr))
        corr[c] = {i: bool(first[i][c]["correct"]) for i in ids}
    print()
    for c in CONDS[1:]:
        x, y, pv = mcn(corr["auto_f@20"], corr[c])
        print("auto_f vs %-9s +%d/-%d p=%.3g" % (c, x, y, pv))
    x, y, pv = mcn(corr["auto_f@20"], corr["full"])
    print("auto_f vs full      +%d/-%d p=%.3g" % (x, y, pv))


if __name__ == "__main__":
    main()
