# -*- coding: utf-8 -*-
"""
抽取失败的归因诊断：错误发生在哪一环？

对每个单步样本，把「gold 答案单元格 (r, c)」与抽取器的选行/选列结果对比：
  - 行选对了吗（r ∈ rows）
  - 列选对了吗（c ∈ cols）
  - 就算行列都对，答案单元格是否真的进了保留集（表格映射缺失？）
并统计**操作数级别**的覆盖——算术任务真正关心的不是答案格，而是全部操作数格。

用法：python diag_recall.py [--limit 300]
"""
import json
import re
import argparse
from collections import Counter

from common import build_dataset
from anchor_auto import extract, table_index
from analyze_slice import nums

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--cross", action="store_true")
    args = ap.parse_args()

    ds = build_dataset(args.data)
    raw = {e.get("id"): e for e in json.load(open(args.data, encoding="utf-8"))}
    cats = Counter()
    shown = {}
    op_miss = Counter()
    n = 0

    for d in ds:
        if n >= args.limit:
            break
        qa = (raw.get(d["id"], {}) or {}).get("qa") or {}
        if not SINGLE.match((qa.get("program") or "").strip()):
            continue
        if not d["group"].get("V"):
            continue
        v = d["group"]["V"][0]
        gr, gc = d["segs"][v]["row"], d["segs"][v]["col"]
        if gr < 0 or gc < 0:
            continue

        idx, diag = extract(d["segs"], d["question"], cross=args.cross)
        cols, rows = set(diag["cols"]), set(diag["rows"])
        ok_r, ok_c = gr in rows, gc in cols
        in_set = v in idx
        cn, un, rh, cell, n_row, n_col = table_index(d["segs"])
        cell_present = (gr, gc) in cell

        if not cell_present:
            cat = "E_表结构缺该格"
        elif ok_r and ok_c and in_set:
            cat = "OK"
        elif ok_r and ok_c:
            cat = "D_行列都对但格未装入"
        elif ok_c and not ok_r:
            cat = "A_选错行"
        elif ok_r and not ok_c:
            cat = "B_选错列"
        else:
            cat = "C_行列都错"

        cats[cat] += 1
        shown.setdefault(cat, [])
        if len(shown[cat]) < 3 and cat != "OK":
            shown[cat].append(dict(
                q=d["question"][:78], gold_cell=d["segs"][v]["text"][:22],
                row_txt=d["segs"][rh[gr]]["text"][:34] if gr in rh else "—",
                col_txt=d["segs"][cn[gc]]["text"][:34] if gc in cn else "—",
                picked_rows=[d["segs"][rh[i]]["text"][:20] for i in list(rows)[:3] if i in rh],
                picked_cols=[d["segs"][cn[j]]["text"][:20] for j in list(cols)[:3] if j in cn],
            ))

        # 操作数级覆盖：程序里的数字落在哪些格，这些格进了保留集吗
        table_nums = set()
        for s in d["segs"]:
            if s["kind"] == "cell":
                table_nums |= nums(s["text"])
        want = nums(qa.get("program") or "") & table_nums
        if want:
            op_cells = []
            for (i, j), si in cell.items():
                if nums(d["segs"][si]["text"]) & want:
                    op_cells.append(si)
            if op_cells:
                hit = sum(1 for si in op_cells if si in idx)
                if hit < len(op_cells):
                    op_miss["操作数少于全部保留"] += 1
                else:
                    op_miss["操作数全保留"] += 1
        n += 1

    tot = sum(cats.values())
    print("样本 n=%d（单步、有 gold 坐标）\n" % tot)
    print("%-24s %6s %7s" % ("类别", "数量", "占比"))
    for k, c in cats.most_common():
        print("%-24s %6d %6.1f%%" % (k, c, 100.0 * c / tot))

    print("\n操作数级:")
    t2 = sum(op_miss.values())
    for k, c in op_miss.most_common():
        print("  %-16s %5d %6.1f%%" % (k, c, 100.0 * c / max(1, t2)))

    for k, v in shown.items():
        print("\n### %s" % k)
        for e in v:
            print("  Q: %s" % e["q"])
            print("     gold 行=%s | 列=%s | 值=%s" % (e["row_txt"], e["col_txt"], e["gold_cell"]))
            print("     选出 行=%s | 列=%s" % (e["picked_rows"], e["picked_cols"]))


if __name__ == "__main__":
    main()
