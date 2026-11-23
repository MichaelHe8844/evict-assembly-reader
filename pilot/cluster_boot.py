# -*- coding: utf-8 -*-
"""按文档聚类的 bootstrap：给主结果加区间，替代"只报点估计 + McNemar"。

同一份财报会产出多道题，题间不独立；McNemar 把每条记录当独立样本，会高估显著性。
这里以**文档**为重采样单位，对"同文档内多条记录"整体抽放。
"""
import json
import random
from collections import defaultdict

random.seed(20261031)
B = 4000


def load(f, cond):
    d = json.load(open(f, encoding="utf-8"))
    out = {}
    for r in d["records"]:
        if r["cond"] != cond:
            continue
        out[(r["id"], round(float(r["gold"]), 6))] = r["correct"]
    return out


def cluster_boot(a, b, label):
    """a=基线，b=处置；按文档重采样，返回 ΔEM 的 95% 区间与双侧 p。"""
    keys = [k for k in b if k in a]
    if not keys:
        print("%-46s 无可配对样本" % label)
        return
    doc = defaultdict(list)
    for k in keys:
        doc[k[0].rsplit("-", 1)[0]].append(k)
    docs = list(doc)
    deltas = []
    for _ in range(B):
        pick = [docs[random.randrange(len(docs))] for _ in docs]
        ks = [k for d in pick for k in doc[d]]
        ca = sum(1 for k in ks if a[k])
        cb = sum(1 for k in ks if b[k])
        deltas.append((cb - ca) / len(ks))
    deltas.sort()
    lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B)]
    na = sum(1 for k in keys if a[k]) / len(keys)
    nb = sum(1 for k in keys if b[k]) / len(keys)
    neg = sum(1 for x in deltas if x <= 0) / B
    pos = sum(1 for x in deltas if x >= 0) / B
    p = min(1.0, 2 * min(neg, pos))
    print("%-46s n=%-4d doc=%-3d  %.3f → %.3f  Δ=%+.3f [%+.3f, %+.3f]  p_boot=%.4f" % (
        label, len(keys), len(docs), na, nb, nb - na, lo, hi, p))


print("按文档聚类 bootstrap（%d 次）\n" % B)

# 真实算术，7B 8-bit，20%
auto = load("../results/cand2_confirm_rescored.json", "auto_f@20")
full = load("../results/cand2_confirm_rescored.json", "full")
snap = load("../results/arith_baselines.json", "snap@20")
h2o = load("../results/arith_baselines.json", "h2o@20")
cluster_boot(snap, auto, "算术 7B：SnapKV@20 → 本文@20")
cluster_boot(h2o, auto, "算术 7B：H2O@20 → 本文@20")
cluster_boot(snap, full, "算术 7B：SnapKV@20 → 全上下文（基线自身的损失）")
cluster_boot(full, auto, "算术 7B：全上下文 → 本文@20（压缩代价）")

# 抽取探针，3B
p_auto = load("../results/xqwen3b_probe.json", "auto_f@20")
p_snap = load("../results/xqwen3b_probe.json", "snap@20")
p_full = load("../results/xqwen3b_probe.json", "full")
cluster_boot(p_snap, p_auto, "探针 3B：SnapKV@20 → 本文@20")
cluster_boot(p_full, p_auto, "探针 3B：全上下文 → 本文@20")

# 留出段，7B
h_auto = load("../results/holdout_arith.json", "auto_f@20")
h_snap = load("../results/holdout_arith.json", "snap@20")
h_full = load("../results/holdout_arith.json", "full")
cluster_boot(h_snap, h_auto, "留出 7B：SnapKV@20 → 本文@20")
cluster_boot(h_full, h_auto, "留出 7B：全上下文 → 本文@20")
