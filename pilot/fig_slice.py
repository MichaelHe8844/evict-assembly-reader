# -*- coding: utf-8 -*-
"""
「保留单元该多大」的三层证据。
  (a) 下游 EM：同一 20% 预算，只改保留单元
  (b) 结构层：该单元能否覆盖算术所需的全部操作数（与模型无关）
  (c) 因果层：操作数是否齐全 → 答对率
"""
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("Microsoft YaHei", "SimHei", "DejaVu Sans"):
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

from common import build_dataset
from arm_c import slice_anchor
from analyze_slice import nums

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")
RES = "../results/arm_c_slice_rescored.json"
UNIT_OF = {"anchor@20": "cell", "row@20": "row", "col@20": "col", "slice@20": "slice"}


def em_close(pred, gold):
    if pred is None:
        return False
    return any(abs(p - gold) <= max(0.01, 0.005 * abs(gold))
               for p in (pred, pred / 100.0, pred * 100.0))


def unit_idx(x, kind):
    if not x["group"].get("V"):
        return None
    v = x["group"]["V"][0]
    s = x["segs"][v]
    if s["row"] < 0 or s["col"] < 0:
        return None
    if kind == "cell":
        return sorted({i for k in "VRCU" for i in x["group"].get(k, [])})
    return slice_anchor(x["segs"], s["row"], s["col"],
                        {"row": "row", "col": "col", "slice": "both"}[kind])


def main():
    d = json.load(open(RES, encoding="utf-8"))
    ds = {x["id"]: x for x in build_dataset("../data/dev.json")}
    RAW = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}

    order = ["full", "anchor@20", "row@20", "col@20", "slice@20"]
    label = {"full": "全上下文", "anchor@20": "单元格", "row@20": "行切片",
             "col@20": "列切片", "slice@20": "十字(行∪列)"}
    ems = {}
    for c in order:
        rs = [r for r in d["records"] if r["cond"] == c]
        if rs:
            ems[c] = float(np.mean([em_close(r["pred"], r["gold"]) for r in rs]))

    # ---- 结构层 + 因果层 ----
    cov_all, cover, correct = {k: [] for k in ("cell", "row", "col", "slice")}, [], []
    for r in d["records"]:
        if r["cond"] not in UNIT_OF:
            continue
        x = ds.get(r["id"])
        if not x:
            continue
        idx = unit_idx(x, UNIT_OF[r["cond"]])
        if idx is None:
            continue
        table = set()
        for s in x["segs"]:
            if s["kind"] == "cell":
                table |= nums(s["text"])
        want = nums(((RAW.get(r["id"], {}).get("qa") or {}).get("program") or "")) & table
        if not want:
            continue
        kept = nums(" ".join(x["segs"][i]["text"] for i in idx))
        f = len(kept & want) / len(want)
        cover.append(f)
        correct.append(em_close(r["pred"], r["gold"]))

    # 结构层用全量 dev 单步样本（不依赖生成结果）
    for e in RAW.values():
        qa = e.get("qa") or {}
        if not SINGLE.match((qa.get("program") or "").strip()):
            continue
        x = ds.get(e.get("id"))
        if not x or not x["group"].get("V"):
            continue
        v = x["group"]["V"][0]
        s = x["segs"][v]
        if s["row"] < 0 or s["col"] < 0:
            continue
        table = set()
        for w in x["segs"]:
            if w["kind"] == "cell":
                table |= nums(w["text"])
        want = nums(qa.get("program") or "") & table
        if not want:
            continue
        for k in ("cell", "row", "col", "slice"):
            idx = unit_idx(x, k)
            kept = nums(" ".join(x["segs"][i]["text"] for i in idx))
            cov_all[k].append(len(kept & want) / len(want))

    cover = np.array(cover)
    correct = np.array(correct)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    # (a) EM
    ax = axes[0]
    cs = [c for c in order if c in ems]
    vals = [ems[c] for c in cs]
    cols = ["#9e9e9e", "#c62828", "#ef6c00", "#1565c0", "#2e7d32"][:len(cs)]
    bars = ax.bar([label[c] for c in cs], vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f" % v,
                ha="center", fontsize=9)
    ax.axhline(ems["full"], ls="--", lw=1, color="#555", alpha=.7)
    ax.set_ylim(0, max(vals) * 1.30 + 0.05)
    ax.set_ylabel("EM")
    ax.set_title("(a) 20% 预算下只改保留单元\n（7B 8-bit，单步算术，n=100）", fontsize=10)
    ax.grid(axis="y", alpha=.25)

    # (b) 结构层
    ax = axes[1]
    keys = ["cell", "row", "col", "slice"]
    names = ["单元格", "行切片", "列切片", "十字"]
    fr = [float((np.array(cov_all[k]) >= 0.999).mean()) for k in keys]
    bars = ax.bar(names, fr, color=["#c62828", "#ef6c00", "#1565c0", "#2e7d32"], width=0.62)
    for b, v in zip(bars, fr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.3f" % v,
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("覆盖全部操作数的样本比例")
    ax.set_title("(b) 该单元能否覆盖算术所需的\n全部操作数（与模型无关，n=%d）"
                 % len(cov_all["cell"]), fontsize=10)
    ax.grid(axis="y", alpha=.25)

    # (c) 因果层
    ax = axes[2]
    full = cover >= 0.999
    a1 = float(correct[full].mean()) if full.any() else float("nan")
    a0 = float(correct[~full].mean()) if (~full).any() else float("nan")
    bars = ax.bar(["操作数齐全", "缺至少一个"], [a1, a0],
                  color=["#2e7d32", "#c62828"], width=0.55)
    for b, v, n in zip(bars, [a1, a0], [int(full.sum()), int((~full).sum())]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f\n(n=%d)" % (v, n),
                ha="center", fontsize=9)
    ax.axhline(ems["full"], ls="--", lw=1, color="#555", alpha=.7)
    ax.set_ylim(0, max(a1, a0) * 1.45 + 0.05)
    ax.set_ylabel("答对率")
    ax.set_title("(c) 操作数齐全与否决定了答对率\n（跨 4 个单元条件合并，n=%d 对）"
                 % len(cover), fontsize=10)
    ax.grid(axis="y", alpha=.25)

    fig.tight_layout()
    fig.savefig("../results/fig_slice.png", dpi=160)
    print("saved ../results/fig_slice.png")
    for c in cs:
        print("  %-10s EM %.3f" % (c, ems[c]))
    print("  结构 n=%d | 因果 P(对|齐)=%.3f P(对|缺)=%.3f  n=%d" % (
        len(cov_all["cell"]), a1, a0, len(cover)))


if __name__ == "__main__":
    main()
