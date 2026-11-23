# -*- coding: utf-8 -*-
"""
预算扫描图：EM 与数值留存率随保留预算的变化。

输入：预算扫描的 arm_c 结果（含 full 与各策略的 @10/20/30/50）。
输出：results/fig_budget.png（双面板）
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

SRC = sys.argv[1] if len(sys.argv) > 1 else "../results/budget_sweep.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../results/fig_budget.png"

d = json.load(open(SRC, encoding="utf-8"))
recs = d["records"]
conds = list(dict.fromkeys(r["cond"] for r in recs))


def soc(c):
    pol, _, bud = c.partition("@")
    return pol, (float(bud) / 100.0 if bud else 1.0)


families = {
    "锚点优先（本文）": ("auto_f", "#c0392b", "o"),
    "SnapKV": ("snap", "#2e86c1", "s"),
    "H2O": ("h2o", "#27ae60", "^"),
    "近期窗口": ("stream", "#7f8c8d", "D"),
}

full_em = np.mean([r["correct"] for r in recs if r["cond"] == "full"])
full_num = np.mean([r["num"] for r in recs if r["cond"] == "full"])

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

for ax, key, ylab, full_v in (
        (axes[0], "correct", "准确率 EM", full_em),
        (axes[1], "num", "数值留存率 NumR", full_num)):
    for name, (pol, color, mk) in families.items():
        xs, ys = [], []
        for c in conds:
            p, frac = soc(c)
            if p != pol or frac >= 1:
                continue
            v = [r[key] for r in recs if r["cond"] == c]
            if v:
                xs.append(frac * 100)
                ys.append(np.mean(v))
        if xs:
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], marker=mk, color=color,
                    label=name, lw=1.6, ms=5)
    ax.axhline(full_v, ls="--", color="black", lw=1.0)
    ax.text(6.5, full_v + 0.03, "全上下文", fontsize=8, ha="left")
    ax.set_xlabel("保留预算（% of document tokens）")
    ax.set_ylabel(ylab)
    ax.set_xlim(5, 55)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_title(ylab.replace("准确率 ", "").replace("数值留存率 ", ""), fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(OUT, dpi=220)
print("写入", OUT)

print("\n%-10s %6s %8s %8s" % ("cond", "n", "EM", "NumR"))
for c in conds:
    v = [r for r in recs if r["cond"] == c]
    if v:
        print("%-10s %6d %8.3f %8.3f" % (
            c, len(v), np.mean([r["correct"] for r in v]), np.mean([r["num"] for r in v])))
