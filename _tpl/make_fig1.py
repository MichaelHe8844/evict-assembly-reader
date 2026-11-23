# -*- coding: utf-8 -*-
"""图 1：三层方法示意（竖排，单栏宽 6.8cm，黑白）。块高按行数自适应。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

F_T = fm.FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=7.2)
F_C = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=7.0)
F_E = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=6.6)

W = 6.8
DARK, MID, LIGHT = "0.12", "0.45", "0.93"
X, BW = 0.10, 6.60
LH = 0.32          # 行距
PAD_T, PAD_B = 0.50, 0.26
GAP = 0.26

BLOCKS = [
    ("线性化财报（压缩前）",
     [("company | payments volume | total transactions", F_E),
      ("american express | 637 | 5.0", F_E)], False),
    ("① 锚点组：保留单元",
     [("把 637 与它的绑定作不可分单元", F_C),
      ("〈V, R, C, U〉", F_C),
      ("V  637    R  american express", F_C),
      ("C  payments volume    U  billions", F_C)], False),
    ("② 定位器：钉住所需一对",
     [("问「每笔交易的平均支付额」", F_C),
      ("→ 选出 637 与 5.0", F_C),
      ("→ 两组合并整组钉入预算", F_C)], False),
    ("③ 程序读者",
     [("只生成程序，算术交执行器", F_C),
      ("divide(637, 5.0) → 127.4", F_E)], False),
    ("可审计的输出",
     [("保留的锚点组 · 选中的一对 · 程序与结果", F_C),
      ("任一环可回查；不能复算则可弃答", F_C)], True),
]

hs = [PAD_T + PAD_B + LH * (len(r) - 1) + 0.10 for _, r, _ in BLOCKS]
total = sum(hs) + GAP * (len(BLOCKS) - 1)
H = total + 0.20

fig = plt.figure(figsize=(W / 2.54, H / 2.54), dpi=500)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

y = H - 0.10
for (title, rows, shade), h in zip(BLOCKS, hs):
    y -= h
    ax.add_patch(FancyBboxPatch((X, y), BW, h,
                                boxstyle="round,pad=0.04,rounding_size=0.10",
                                linewidth=0.8, edgecolor=MID,
                                facecolor=LIGHT if shade else "white"))
    ax.text(X + 0.16, y + h - 0.10, title, ha="left", va="top", fontproperties=F_T, color=DARK)
    yy = y + h - PAD_T
    for r, f in rows:
        ax.text(X + 0.16, yy, r, ha="left", va="top", fontproperties=f, color=DARK)
        yy -= LH
    if y > 0.12:
        ax.add_patch(FancyArrowPatch((X + BW / 2, y), (X + BW / 2, y - GAP + 0.02),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color=MID))
    y -= GAP

fig.savefig("results/fig_method.png", dpi=500, facecolor="white")
print("ok 高 %.2f cm" % H)
