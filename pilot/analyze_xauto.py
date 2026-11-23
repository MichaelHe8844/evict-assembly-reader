# -*- coding: utf-8 -*-
"""
端到端对照：无监督抽取，单元格单元（auto_f）vs 十字单元（auto_x）。
同一个抽取器、同一个问题、同一 20% 预算，只改装配后的单元形状。
"""
import json
import re
import sys
import numpy as np
from math import comb

from common import build_dataset
from anchor_auto import extract
from analyze_slice import nums


def em_close(pred, gold):
    if pred is None:
        return False
    return any(abs(p - gold) <= max(0.01, 0.005 * abs(gold))
               for p in (pred, pred / 100.0, pred * 100.0))


def mcnemar(a, b):
    x = int((a & ~b).sum())
    y = int((b & ~a).sum())
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/arm_c_xauto.json"
    d = json.load(open(path, encoding="utf-8"))
    conds = []
    for r in d["records"]:
        if r["cond"] not in conds:
            conds.append(r["cond"])

    print("== EM（尺度归一）==")
    seq = {c: [r for r in d["records"] if r["cond"] == c] for c in conds}
    n = min(len(v) for v in seq.values())
    ok = {}
    for c in conds:
        v = np.array([em_close(seq[c][i]["pred"], seq[c][i]["gold"]) for i in range(n)])
        ok[c] = v
        print("  %-12s n=%d  EM %.3f" % (c, len(seq[c]), v.mean()))

    print("\n== 配对 McNemar（n=%d）==" % n)
    base = "auto_f@20" if "auto_f@20" in ok else conds[0]
    for c in conds:
        if c == base:
            continue
        x, y, p = mcnemar(ok[base], ok[c])
        print("  %-12s vs %-12s: 前者独对 %2d | 后者独对 %2d  p=%.3f"
              % (base, c, x, y, p))

    # 结构层的覆盖对比（0.810 vs 0.875）在 200 个独立单步样本上单独测过，
    # 见论文 5.5 节。这里只做生成侧对照，避免按 id 回查导致问题错配
    # （FinQA 的 id 是「文件-页」，同页多问题共享 id）。


if __name__ == "__main__":
    main()
