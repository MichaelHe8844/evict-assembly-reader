# -*- coding: utf-8 -*-
"""跨模型族复核：同一数据、同一 20% 预算、同一脚手架，只换模型。

输出三张表：
  表 A 各条件 EM / NumR / ANR / DGR（数字级留存）
  表 B 把「预算比例」换成「数字留存」后，各族是否落在同一条曲线上
  表 C Llama 的预算扫描（同族内也服从 DGR，说明不是族特异）
"""
import json
import numpy as np

MODELS = [("Qwen2.5-3B", "results/xqwen3b_probe.json"),
          ("Qwen2.5-7B", "results/xqwen7b_probe.json"),
          ("Llama-3.2-3B", "results/xllama_probe.json"),
          ("Phi-3.5-mini", "results/xphi35_probe.json")]
CONDS = ["full", "snap@20", "anchor@20", "auto_f@20", "auto_n@20"]


def agg(path, cond):
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in d["records"] if r["cond"] == cond]
    if not rs:
        return None
    dg = np.array([r.get("dgr", np.nan) for r in rs], dtype=float)
    ok = np.array([r["correct"] for r in rs], dtype=float)
    return dict(n=len(rs), em=ok.mean(), numr=np.mean([r["num"] for r in rs]),
                anr=np.mean([r["anr"] for r in rs]),
                dgr=float(np.nanmean(dg)), dgr_full=float((dg > 0.999).mean()),
                tok=float(np.mean([r["n_doc"] for r in rs])),
                acc_dgr1=float(ok[dg > 0.999].mean()) if (dg > 0.999).any() else float("nan"),
                n_dgr1=int((dg > 0.999).sum()))


print("表 A  各条件（150 探针 / 78 文档）")
print("%-13s %-10s %6s %6s %6s %6s %6s" % ("model", "cond", "EM", "NumR", "ANR", "DGR", "P(DGR=1)"))
rows = {}
for lab, path in MODELS:
    for c in CONDS:
        a = agg(path, c)
        if not a:
            continue
        rows[(lab, c)] = a
        print("%-13s %-10s %6.3f %6.3f %6.3f %6.3f %6.3f" % (
            lab, c, a["em"], a["numr"], a["anr"], a["dgr"], a["dgr_full"]))
    print()

print("表 B  按 digital retention 分档（三族合并，非 full 条件）")
pool = []
for lab, path in MODELS:
    d = json.load(open(path, encoding="utf-8"))
    for r in d["records"]:
        if r["cond"] == "full":
            continue
        pool.append((r.get("dgr", float("nan")), r["correct"]))
dg = np.array([p[0] for p in pool], dtype=float)
ok = np.array([p[1] for p in pool], dtype=float)
for lo, hi, name in [(0, 1e-9, "DGR=0"), (1e-9, 0.6, "0<DGR<0.6"),
                     (0.6, 0.999, "0.6-1.0"), (0.999, 9, "DGR=1")]:
    m = (dg >= lo) & (dg < hi)
    if m.sum():
        print("  %-10s n=%4d  acc=%.3f" % (name, int(m.sum()), ok[m].mean()))

print("\n表 C  Llama-3.2-3B 预算扫描（DGR 而不是 token 比例才是可比的横轴）")
for c in ["snap@5", "snap@7", "snap@10", "snap@20", "auto_f@20"]:
    a = agg("results/xllama_sweep.json", c) or agg("results/xllama_probe.json", c)
    if not a:
        continue
    print("  %-10s EM=%.3f NumR=%.3f DGR=%.3f tokens=%.0f" % (
        c, a["em"], a["numr"], a["dgr"], a["tok"]))

json.dump({("%s|%s" % k): v for k, v in rows.items()},
          open("results/xmodel_summary.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n写入 results/xmodel_summary.json")
