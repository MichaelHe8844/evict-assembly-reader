# -*- coding: utf-8 -*-
"""图：缓存的信息上界与读者的兑现率（两个上界）。

数据来自预实验报告 3.42：87 道有 gold 对的 dev 题，
检查 gold 两个操作数的数值是否同时仍在保留文本里。
OOS 只在 20% 档有值（其预测来自 20% 池）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

B = ["3%", "5%", "20%"]
ceil_ = [0.402, 0.506, 0.529]
em3b = [0.345, 0.379, 0.379]
rl3b = [0.857, 0.750, 0.717]
em_oos, rl_oos = 0.552, 0.552 / 0.529

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))

x = np.arange(len(B))
w = 0.36
a = ax[0]
a.bar(x - w / 2, ceil_, w, label="文本通道上界（两数都在）", color="#8fb8dc", edgecolor="black", linewidth=0.4)
a.bar(x + w / 2, em3b, w, label="3B 实际 EM", color="#d98b5f", edgecolor="black", linewidth=0.4)
a.plot([2], [em_oos], marker="*", ms=9, color="#2f6f3e", ls="none", label="结构化通道（20%）")
for i, (c, e) in enumerate(zip(ceil_, em3b)):
    a.text(i - w / 2, c + 0.012, "%.2f" % c, ha="center", fontsize=7.5)
    a.text(i + w / 2, e + 0.012, "%.2f" % e, ha="center", fontsize=7.5)
a.text(2, em_oos + 0.03, "%.2f" % em_oos, ha="center", fontsize=7.5, color="#2f6f3e")
a.set_xticks(x); a.set_xticklabels(B)
a.set_ylim(0, 0.66); a.set_ylabel("比例")
a.set_xlabel("保留预算（占文档 token）")
a.set_title("(a) 上界与实际准确率", fontsize=9)
a.legend(fontsize=7, loc="upper left", frameon=False)
a.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

b = ax[1]
b.plot(x, rl3b, marker="o", ms=5, color="#c0504d", label="3B 兑现率")
b.plot([2], [rl_oos], marker="*", ms=10, color="#2f6f3e", ls="none", label="结构化通道（20%）")
for i, v in enumerate(rl3b):
    b.text(i, v + 0.02, "%.1f%%" % (100 * v), ha="center", fontsize=7.5)
b.text(2, rl_oos + 0.02, "%.1f%%" % (100 * rl_oos), ha="center", fontsize=7.5, color="#2f6f3e")
b.set_xticks(x); b.set_xticklabels(B)
b.set_ylim(0.62, 1.12); b.set_ylabel("兑现率 = EM / 信息上界")
b.set_xlabel("保留预算（占文档 token）")
b.set_title("(b) 读者把\"有的\"用掉了多少", fontsize=9)
b.legend(fontsize=7, loc="lower left", frameon=False)
b.axhline(1.0, ls="--", lw=0.8, color="#666666")
b.text(0.02, 1.015, "文本通道上界", fontsize=7, color="#666666")
b.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("../results/fig_ceiling.png", dpi=300)
print("saved ../results/fig_ceiling.png")
