# -*- coding: utf-8 -*-
"""比较「锚点装配」与「锚点装配 + 选择器强制保留」的端到端效果。

同时报三件容易混淆的事：
  1. EM 与配对检验（同题、同预算、同模型）
  2. 实际保留 token 数（强制保留会略微超出预算，必须如实报）
  3. 覆盖率与 EM 的关系（选中的两格是否真被保住，不等于答对）
"""
import json
import math
import sys

import numpy as np


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
    p_base, p_sel = sys.argv[1], sys.argv[2]
    cond = sys.argv[3] if len(sys.argv) > 3 else "auto_f@5"
    A, B = load(p_base, cond), load(p_sel, cond)
    ks = [k for k in B if k in A]
    a = {k: bool(A[k]["correct"]) for k in ks}
    b = {k: bool(B[k]["correct"]) for k in ks}
    print("配对样本 %d" % len(ks))
    print("EM：基线 %.3f → 强制保留 %.3f（%+.3f）"
          % (np.mean(list(a.values())), np.mean(list(b.values())),
             np.mean(list(b.values())) - np.mean(list(a.values()))))
    x, y, p = mcn(b, a)
    print("  +%d / -%d，McNemar p=%.4f" % (x, y, p))
    for name, D in (("基线", A), ("强制保留", B)):
        rec = [D[k] for k in ks]
        print("  %s：ANR %.3f | 保留 token %.0f（预算 %.0f）| L %.0f"
              % (name, np.mean([r["anr"] for r in rec]),
                 np.mean([r["budget_tokens"] for r in rec]),
                 np.mean([int(0.05 * r["n_doc"]) for r in rec]),
                 np.mean([r["L"] for r in rec])))
    # 强制保留带来的 token 增量
    db = np.array([B[k]["budget_tokens"] - A[k]["budget_tokens"] for k in ks])
    print("保留 token 增量：均值 %+.1f，中位 %+.0f，最大 %+d"
          % (db.mean(), np.median(db), db.max()))


if __name__ == "__main__":
    main()
