# -*- coding: utf-8 -*-
"""
检验「保留单元该多大」：单元格 / 行切片 / 列切片。

三个层次，由外到内：
  1. EM：下游准确率（需生成结果）
  2. 操作数覆盖率：保留的片段里出现了程序所需的几个操作数（与模型无关，纯结构）
  3. 预算占用：保留了多少 token

第 2 项是关键：它把「为什么 cell 级不够」从猜测变成可验证的结构事实。
"""
import json
import re
import sys
import numpy as np

from common import build_dataset
from arm_c import slice_anchor

NUM = re.compile(r"\d[\d,]*\.?\d*")


def nums(s):
    out = set()
    for m in NUM.findall(s or ""):
        t = m.replace(",", "").rstrip(".")
        try:
            v = float(t)
        except ValueError:
            continue
        out.add(round(v, 6))
    return out


def em_close(pred, gold):
    if pred is None:
        return False
    for p in (pred, pred / 100.0, pred * 100.0):
        if abs(p - gold) <= max(0.01, 0.005 * abs(gold)):
            return True
    return False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/arm_c_slice.json"
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]
    ds = {x["id"]: x for x in build_dataset("../data/dev.json")}
    RAW = {e.get("id"): e for e in json.load(open("../data/dev.json", encoding="utf-8"))}

    conds = []
    for r in recs:
        if r["cond"] not in conds:
            conds.append(r["cond"])

    print("== 1. EM（尺度归一，允许 $ 与逗号）==")
    print("%-11s %5s %6s %8s %8s" % ("cond", "n", "parse", "EM", "EM_strict"))
    for c in conds:
        rs = [r for r in recs if r["cond"] == c]
        ok = [em_close(r["pred"], r["gold"]) for r in rs]
        st = [r["pred"] is not None and abs(r["pred"] - r["gold"]) <=
              max(0.01, 0.005 * abs(r["gold"])) for r in rs]
        print("%-11s %5d %6.3f %8.3f %8.3f" % (
            c, len(rs), np.mean([r["pred"] is not None for r in rs]),
            np.mean(ok), np.mean(st)))

    # ---- 2/3. 结构层：保留单元覆盖了哪些操作数 ----
    print("\n== 2. 保留单元的操作数覆盖（与模型无关）==")
    stat = {k: [] for k in ("cell", "row", "col", "slice")}
    budget = {k: [] for k in ("cell", "row", "col", "slice")}

    seen = set()
    for r in recs:
        sid = r["id"]
        if sid in seen:
            continue
        seen.add(sid)
        x = ds.get(sid)
        if not x or not x["group"].get("V"):
            continue
        raw = x.get("question") or ""
        v = x["group"]["V"][0]
        ar, ac = x["segs"][v]["row"], x["segs"][v]["col"]

        # 表中出现的数字（候选操作数）
        table_nums = set()
        for s in x["segs"]:
            if s["kind"] == "cell":
                table_nums |= nums(s["text"])
        prog = ""
        for rr in recs:
            if rr["id"] == sid:
                prog = rr.get("program", "") or ""
                break
        if not prog:
            raw_e = RAW.get(sid) or {}
            prog = ((raw_e.get("qa") or {}).get("program") or "")
        want = nums(prog) & table_nums if prog else set()

        units = {
            "cell": sorted({i for k in "VRCU" for i in x["group"].get(k, [])}),
            "row": slice_anchor(x["segs"], ar, ac, "row"),
            "col": slice_anchor(x["segs"], ar, ac, "col"),
            "slice": slice_anchor(x["segs"], ar, ac, "both"),
        }
        for k, idx in units.items():
            kept = nums(" ".join(x["segs"][i]["text"] for i in idx))
            if want:
                stat[k].append(len(kept & want) / len(want))
            budget[k].append(len(idx))

    if not any(stat.values()):
        print("  （结果文件未保存 program，跳过）")
    else:
        print("%-11s %6s %8s %8s" % ("unit", "n", "全操作数", "平均覆盖"))
        for k in ("cell", "row", "col", "slice"):
            a = np.array(stat[k])
            print("%-11s %6d %8.3f %8.3f" % (
                k, len(a), float((a >= 0.999).mean()), float(a.mean())))

    print("\n== 3. 保留单元的片段数（预算粗略代理）==")
    for k in ("cell", "row", "col", "slice"):
        a = np.array(budget[k])
        print("%-11s median %.0f  p90 %.0f  max %.0f" % (
            k, np.median(a), np.percentile(a, 90), a.max()))

    # ---- 4. 配对比较：cell 失败但 slice 成功的样本 ----
    # 注意：FinQA 的 id 是「文件-页」，同一页有多个 QA，id 不唯一。
    # 因此按「同一条件下第 k 条记录」配对，而不是按 id。
    print("\n== 4. 配对比较（同一样本跨条件）==")
    seq = {}
    for r in recs:
        seq.setdefault(r["cond"], []).append(r)
    n_pair = min((len(v) for v in seq.values()), default=0)
    okm = {c: np.array([em_close(v[i]["pred"], v[i]["gold"]) for i in range(n_pair)])
           for c, v in seq.items()}
    for a, b in (("anchor@20", "row@20"), ("anchor@20", "col@20"),
                 ("anchor@20", "slice@20"), ("row@20", "slice@20"),
                 ("full", "slice@20")):
        if a not in okm or b not in okm:
            continue
        both = int((okm[a] & okm[b]).sum())
        only_a = int((okm[a] & ~okm[b]).sum())
        only_b = int((okm[b] & ~okm[a]).sum())
        print("%-20s n=%d  两者都对 %d | 只有前者 %d | 只有后者 %d" % (
            "%s vs %s" % (a, b), n_pair, both, only_a, only_b))


if __name__ == "__main__":
    main()
