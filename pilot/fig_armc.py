# -*- coding: utf-8 -*-
"""Arm C 的出图：条件准确率 + 锚点完整/碎裂的条件准确率。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("../results/arm_c.json", encoding="utf-8"))
S = {r["cond"]: r for r in d["summary"]}

order = ["full", "snap@20", "snap@50", "stream@20", "anchor@20"]
label = ["Full\ncontext", "SnapKV\n20%", "SnapKV\n50%", "Recency\n20%", "Anchor-first\n20% (oracle)"]
color = ["#9aa0a6", "#f2a154", "#e8743b", "#7fb3d5", "#2e8b57"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# 左：各条件 EM
ax = axes[0]
vals = [S[c]["em"] for c in order]
bars = ax.bar(range(len(order)), vals, color=color, width=.62)
for i, v in enumerate(vals):
    ax.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(label, fontsize=8)
ax.set_ylabel("Exact match")
ax.set_ylim(0, 1.0)
ax.set_title("Same document, different retention policies", fontsize=10)
ax.grid(axis="y", alpha=.3)
ax.axhline(S["full"]["em"], ls=":", color="gray", lw=1)

# 右：同预算下 锚点完整 vs 碎裂
ax = axes[1]
conds = ["snap@20", "snap@50", "stream@20"]
x = np.arange(len(conds))
w = .36
v1 = [S[c]["p_correct_intact"] for c in conds]
v0 = [S[c]["p_correct_frag"] for c in conds]
ax.bar(x - w / 2, v1, w, label="anchor group intact", color="#2e8b57")
ax.bar(x + w / 2, v0, w, label="anchor group broken", color="#c0392b")
for i in range(len(conds)):
    ax.text(i - w / 2, v1[i] + .02, f"{v1[i]:.2f}", ha="center", fontsize=8.5)
    ax.text(i + w / 2, v0[i] + .02, f"{v0[i]:.2f}", ha="center", fontsize=8.5)
    ax.text(i, -.14, f"n={S[conds[i]]['n_intact']}/{S[conds[i]]['n_frag']}",
            ha="center", fontsize=7.5, color="gray")
ax.set_xticks(x)
ax.set_xticklabels([l.replace("\n", " ") for l in ["SnapKV 20%", "SnapKV 50%", "Recency 20%"]], fontsize=8.5)
ax.set_ylabel("Exact match")
ax.set_ylim(0, 1.0)
ax.set_title("Within the same budget, binding decides accuracy", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=.3)

fig.suptitle("KV-cache eviction breaks number–context binding and the answer with it "
             "(FinQA dev, Qwen2.5-3B, cell-extraction probe)", fontsize=10.5)
fig.tight_layout()
fig.savefig("../results/fig_armc.png", dpi=170)
print("saved fig_armc.png")

print("\nEM 对比（同预算 20%%）: anchor %.3f vs snap %.3f vs recency %.3f  (full %.3f)" % (
    S["anchor@20"]["em"], S["snap@20"]["em"], S["stream@20"]["em"], S["full"]["em"]))
