# -*- coding: utf-8 -*-
"""汇总 Arm A / Arm B，出表出图。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "../results"


def load(p):
    return json.load(open(p, encoding="utf-8"))


a = load(f"{RES}/arm_a.json")
b_ag = load(f"{RES}/arm_b_agnostic.json")
b_aw = load(f"{RES}/arm_b_aware.json")

rows = []
for r in a["rows"]:
    rows.append(dict(exp="A", **r))
for tag, d in (("B-agnostic", b_ag), ("B-aware", b_aw)):
    for r in d["rows"]:
        rows.append(dict(exp=tag, **r))

# ---------- 图 1：NumR / ANR 随预算变化 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
series = {
    "StreamingLLM (recency)": ("A", "streaming", "tab:blue"),
    "Head+Tail truncation": ("A", "head_tail", "tab:cyan"),
    "SnapKV (query-agnostic)": ("B-agnostic", "snapkv", "tab:orange"),
    "SnapKV (query-aware)": ("B-aware", "snapkv", "tab:red"),
}
for ax, key, title in ((axes[0], "NumR", "Number cell kept intact (NumR)"),
                       (axes[1], "ANR", "Full anchor group kept (ANR)")):
    for label, (exp, pol, color) in series.items():
        pts = [r for r in rows if r["exp"] == exp and r["policy"] == pol]
        if not pts:
            continue
        pts.sort(key=lambda r: r["budget"])
        ax.plot([p["budget"] * 100 for p in pts], [p[key] for p in pts],
                marker="o", ms=4, lw=1.8, label=label, color=color)
    # oracle 上界
    pts = [r for r in rows if r["exp"] == "B-aware" and r["policy"] == "oracle"]
    pts.sort(key=lambda r: r["budget"])
    ax.plot([p["budget"] * 100 for p in pts], [p[key] for p in pts],
            ls="--", lw=1.5, color="tab:green", label="Anchor-first (oracle)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("KV cache budget (% of context)")
    ax.grid(alpha=.3)
    ax.set_ylim(-0.02, 1.05)
axes[0].set_ylabel("retention rate")
axes[1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Financial numeric anchors under KV-cache compression (FinQA dev, Qwen2.5-3B)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{RES}/fig_retention.png", dpi=170)
print("saved fig_retention.png")

# ---------- 图 2：注意力富集比 ----------
fig2, ax = plt.subplots(figsize=(5.6, 3.6))
comps = ["BG", "V", "R", "C", "U"]
labels = ["background\n(non-anchor)", "value\n(V)", "row header\n(R)", "column name\n(C)", "unit\n(U)"]
x = np.arange(len(comps))
w = 0.38
for i, (tag, d, c) in enumerate((("query-agnostic", b_ag, "tab:orange"),
                                 ("query-aware", b_aw, "tab:red"))):
    vals = [d["enrich"].get(k, np.nan) for k in comps]
    ax.bar(x + (i - 0.5) * w, vals, w, label=tag, color=c, alpha=.85)
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("attention enrichment\n(share of attention / share of tokens)")
ax.set_title("Numbers are an attention trough, not a peak", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=.3)
fig2.tight_layout()
fig2.savefig(f"{RES}/fig_enrichment.png", dpi=170)
print("saved fig_enrichment.png")

# ---------- CSV ----------
import csv
with open(f"{RES}/summary.csv", "w", newline="", encoding="utf-8") as f:
    wtr = csv.DictWriter(f, fieldnames=["exp", "policy", "budget", "NumR", "ANR",
                                        "R", "C", "U", "frag_cond"])
    wtr.writeheader()
    for r in rows:
        wtr.writerow({k: r.get(k) for k in wtr.fieldnames})
print("saved summary.csv")

# ---------- 文本摘要 ----------
print("\n=== 关键行 ===")
for r in rows:
    if r["budget"] in (0.2, 0.5):
        print(f'{r["exp"]:>11} {r["policy"]:>9} {r["budget"]*100:>3.0f}%  '
              f'NumR {r["NumR"]:.3f}  ANR {r["ANR"]:.3f}  frag|V {r["frag_cond"]:.3f}')
