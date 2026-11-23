# -*- coding: utf-8 -*-
"""
把真实算术任务上的基线（SnapKV/H2O）与已有条件（full/auto_f/auto_n）合并成一张主结果表。

两个结果文件必须来自同一批样本：按 id 对齐并校验，若顺序不一致则报错。
"""
import json
import sys
from math import comb

import numpy as np

BASE = "../results/cand2_confirm_rescored.json"
NEW = sys.argv[1] if len(sys.argv) > 1 else "../results/arith_baselines_rescored.json"


def load(p, conds):
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    for r in d["records"]:
        if r["cond"] in conds:
            out.setdefault(r["cond"], []).append(r)
    return out


def main():
    base = load(BASE, {"full", "auto_f@20", "auto_n@20"})
    new = load(NEW, {"snap@20", "h2o@20"})

    ref = [r["id"] for r in base["full"]]
    ok = True
    for c, rs in new.items():
        ids = [r["id"] for r in rs]
        if ids != ref:
            ok = False
            n = min(len(ids), len(ref))
            print("WARN %s 与基准样本顺序不一致（重合 %d/%d）" % (c, sum(
                a == b for a, b in zip(ids, ref)), n))
    if not ok:
        print("按 id 重新对齐……")
        idx = {i: k for k, i in enumerate(ref)}
        for c in list(new):
            new[c] = sorted([r for r in new[c] if r["id"] in idx],
                            key=lambda r: idx[r["id"]])

    n = len(ref)
    print("样本 n=%d\n" % n)
    print("%-22s %8s %8s %s" % ("条件", "EM", "ANR", "95% CI"))
    res = {}
    order = [("全上下文", "full", base), ("SnapKV 20%", "snap@20", new),
             ("H2O 20%", "h2o@20", new), ("本文（无监督）", "auto_f@20", base),
             ("本文（关闭亚军候选）", "auto_n@20", base)]
    rng = np.random.default_rng(0)
    for name, c, src in order:
        if c not in src:
            print("%-22s 缺失" % name)
            continue
        v = np.array([r["correct"] for r in src[c][:n]])
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(3000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        anr = np.mean([r["anr"] for r in src[c][:n]])
        res[c] = v
        print("%-22s %8.3f %8.3f [%.2f, %.2f]" % (name, v.mean(), anr, lo, hi))

    def mcn(a, b, na, nb):
        x = int((a & ~b).sum())   # a 对 b 错
        y = int((b & ~a).sum())
        m = x + y
        if not m:
            return x, y, 1.0
        k = min(x, y)
        return x, y, min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)

    print("\n配对检验（vs 本文）:")
    for c in ("snap@20", "h2o@20", "full"):
        if c in res:
            x, y, p = mcn(res["auto_f@20"], res[c], "本文", c)
            print("  本文独对 %2d | %-9s 独对 %2d | p=%.4f" % (x, c, y, p))


if __name__ == "__main__":
    main()
