# -*- coding: utf-8 -*-
"""无监督锚点抽取：准确率与召回率的分解图（改进前后对比）。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = json.load(open("../results/arm_c_final.json", encoding="utf-8"))
S = {r["cond"]: r for r in d["summary"]}
E = json.load(open("../results/anchor_auto_eval2.json", encoding="utf-8"))["finqa"]

order = ["full", "snap@20", "snap@50", "stream@20", "anchor@20", "auto_f@20"]
order = [c for c in order if c in S]
label_map = {
    "full": "Full\ncontext", "snap@20": "SnapKV\n20%", "snap@50": "SnapKV\n50%",
    "stream@20": "Recency\n20%", "anchor@20": "Anchor-first\n20% (gold)",
    "auto_f@20": "Anchor-first\n20% (auto)",
}
color_map = {
    "full": "#9aa0a6", "snap@20": "#f2a154", "snap@50": "#e8743b",
    "stream@20": "#7fb3d5", "anchor@20": "#2e8b57", "auto_f@20": "#1f5f3f",
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

ax = axes[0]
vals = [S[c]["em"] for c in order]
ax.bar(range(len(order)), vals, color=[color_map[c] for c in order], width=.64)
for i, v in enumerate(vals):
    ax.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([label_map[c] for c in order], fontsize=7.6)
ax.set_ylabel("Exact match")
ax.set_ylim(0, 1.0)
ax.axhline(S["full"]["em"], ls=":", color="gray", lw=1)
ax.set_title("Same 20%% budget: %.3f with unsupervised anchors" % S["auto_f@20"]["em"],
             fontsize=10)
ax.grid(axis="y", alpha=.3)

ax = axes[1]
comps = ["V", "R", "C", "U"]
names = ["value\ncell", "row\nheader", "column\nname", "unit"]
before = [0.727, 0.940, 0.757, 1.000]      # 首版：词面匹配 + 字面回退
after = [E["comp"]["V"], E["comp"]["R"], E["comp"]["C"], E["comp"]["U"]]
x = np.arange(4)
w = .36
ax.bar(x - w / 2, before, w, label="first version", color="#b0b7bd")
ax.bar(x + w / 2, after, w, label="improved", color="#2e8b57")
for i in range(4):
    ax.text(i - w / 2, before[i] + .015, f"{before[i]:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, after[i] + .015, f"{after[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0, 1.3)
ax.set_ylabel("per-sample full hit rate")
ax.set_title("Year match + fuzzy match + column hedging", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=.3)
ax.text(0.5, 0.955,
        "full recall 0.727 → %.3f ｜ cost 5.4%% → %.1f%%" % (
            E["full_recall_rate"], 100 * E["cost_tok"]),
        transform=ax.transAxes, ha="center", fontsize=8, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bbb"))

fig.suptitle("Unsupervised anchor extraction: recall %.3f, %.3f EM at 20%% budget "
             "(FinQA dev, Qwen2.5-3B)" % (E["full_recall_rate"], S["auto_f@20"]["em"]),
             fontsize=10.5)
fig.tight_layout()
fig.savefig("../results/fig_auto.png", dpi=170)
print("saved fig_auto.png")

r = E["full_recall_rate"]
anr = S["auto_f@20"]["anr"]
p1, p0 = S["auto_f@20"]["p_correct_intact"], S["auto_f@20"]["p_correct_frag"]
print("\n抽取完整召回 %.3f → 落地 ANR %.3f（差额来自 20%% 预算硬约束）" % (r, anr))
print("分解: %.3f*%.3f + %.3f*%.3f = %.3f（实测 %.3f）" % (
    anr, p1, 1 - anr, p0, anr * p1 + (1 - anr) * p0, S["auto_f@20"]["em"]))
for rr in (0.95, 0.98):
    print("  若抽取召回 %.2f -> EM ≈ %.3f" % (rr, rr * p1 + (1 - rr) * p0))
