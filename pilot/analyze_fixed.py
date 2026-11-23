# -*- coding: utf-8 -*-
"""严格同预算下的强制保留：主表 + 回退分解。

与旧口径的区别：旧版把选中的一对「额外」塞进保留集，实际 token 超出预算
（5% 档平均 +10.5）；本脚本比较的是 --budget_mode fixed（总 token 恰为 B）。

用法：python analyze_fixed.py
依赖：
  dev  ../results/e2e_af{3,5,10,20}_base.json  vs ../results/fix_af{3,5,10,20}.json
  test ../results/x{3,5,10}_base.json          vs ../results/fix_x{3,5,10}.json
"""
import json
import math

import numpy as np


def load(path, cond):
    d = json.load(open(path, encoding="utf-8"))
    return {(r["id"], round(float(r["gold"]), 6)): r
            for r in d["records"] if r["cond"] == cond}


def mcn(a, b):
    """a=基线，b=处理。返回 (处理对/基线对, 处理错/基线错, p)。"""
    x = sum(1 for k in a if b[k] and not a[k])
    y = sum(1 for k in a if a[k] and not b[k])
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def one(a_path, b_path, cond, tag):
    try:
        A, B = load(a_path, cond), load(b_path, cond)
    except FileNotFoundError:
        print("%-10s （缺数据）" % tag)
        return
    ks = [k for k in B if k in A]
    a = {k: bool(A[k]["correct"]) for k in ks}
    b = {k: bool(B[k]["correct"]) for k in ks}
    x, y, p = mcn(a, b)
    tb = np.mean([B[k]["budget_tokens"] for k in ks])
    ta = np.mean([A[k]["budget_tokens"] for k in ks])
    fit = {c: sum(1 for k in ks if B[k].get("pair_fit") == c) for c in (0, 1, 2, 3)}
    n = len(ks)
    print("%-9s %4d  %6.3f  %6.3f  %+6.3f   +%-3d/-%-3d %8.1e   %.1f/%.1f   %5.1f%%   %4.1f/%4.1f/%4.1f"
          % (tag, n, np.mean(list(a.values())), np.mean(list(b.values())),
             np.mean(list(b.values())) - np.mean(list(a.values())),
             x, y, p, ta, tb, 100.0 * fit[0] / n,
             100.0 * fit[1] / n, 100.0 * fit[2] / n, 100.0 * fit[3] / n))
    # 按回退码分层：只有「整对装入」这一层可能受益
    for c, name in ((0, "  整对装入"), (1, "  仅两格数值"), (2, "  仅一格"), (3, "  放弃")):
        g = [k for k in ks if B[k].get("pair_fit") == c]
        if not g:
            continue
        ga = np.mean([a[k] for k in g])
        gb = np.mean([b[k] for k in g])
        print("            %s n=%-3d EM %.3f → %.3f（%+.3f）"
              % (name, len(g), ga, gb, gb - ga))


def main():
    print("%-9s %4s  %6s  %6s  %6s   %-9s %8s   %-9s   %-5s   %s"
          % ("预算", "n", "基线", "固定", "Δ", "配对", "p", "token 基线/固定",
             "整对装入", "回退% 数值/单格/放弃"))
    for b in (3, 5, 10, 20):
        one("../results/e2e_af%d_base.json" % b, "../results/fix_af%d.json" % b,
            "auto_f@%d" % b, "dev %d%%" % b)
    for b in (3, 5, 10):
        one("../results/x%d_base.json" % b, "../results/fix_x%d.json" % b,
            "auto_f@%d" % b, "test %d%%" % b)


if __name__ == "__main__":
    main()
