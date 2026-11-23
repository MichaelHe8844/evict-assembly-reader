# -*- coding: utf-8 -*-
"""图：预算-准确率交叉（编译式读者 vs 语言模型），dev 与 test 双面板。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

DEV = dict(x=[3, 5, 10, 20],
           cmp=[0.048, 0.214, 0.500, 0.587],
           lm=[0.270, 0.294, 0.333, 0.317])
TEST = dict(x=[3, 5, 10, 20],
            cmp=[0.154, 0.269, 0.489, 0.599],
            lm=[0.319, 0.451, 0.484, 0.473])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, D, name, n in ((axes[0], DEV, "(a) dev", 126),
                       (axes[1], TEST, "(b) 官方 test", 182)):
    xs = list(range(len(D["x"])))
    ax.plot(xs, D["cmp"], "o-", color="#C0392B", lw=1.6, ms=4.5, label="编译式读者")
    ax.plot(xs, D["lm"], "s--", color="#2C6E9B", lw=1.6, ms=4.5, label="语言模型（同底座）")
    ax.fill_between(xs, D["cmp"], D["lm"],
                    where=[c > l for c, l in zip(D["cmp"], D["lm"])],
                    color="#C0392B", alpha=0.08, interpolate=True)
    ax.fill_between(xs, D["cmp"], D["lm"],
                    where=[c < l for c, l in zip(D["cmp"], D["lm"])],
                    color="#2C6E9B", alpha=0.08, interpolate=True)
    ax.set_xticks(range(len(D["x"])))
    ax.set_xticklabels(["%d%%" % v for v in D["x"]])
    ax.minorticks_off()
    ax.set_xlabel("证据预算")
    ax.set_title("%s，%d 题" % (name, n), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(0, 0.74)
    for i, c, l in zip(xs, D["cmp"], D["lm"]):
        near = abs(c - l) < 0.09
        ax.annotate("%.2f" % c, (i, c), textcoords="offset points",
                    xytext=(0, 11 if near else 7),
                    ha="center", fontsize=6.8, color="#C0392B")
        ax.annotate("%.2f" % l, (i, l), textcoords="offset points",
                    xytext=(0, -16 if near else -12),
                    ha="center", fontsize=6.8, color="#2C6E9B")
axes[0].set_ylabel("EM")
axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
plt.tight_layout()
plt.savefig("../results/fig_budget.png", dpi=200)
print("saved")
