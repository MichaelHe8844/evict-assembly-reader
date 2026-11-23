# -*- coding: utf-8 -*-
"""
多表闭包的结构验证：把「每个操作数所在表各取一个十字」真正装配出来，
对比 单元格 / 单十字 / 多表闭包 三种单元对操作数的覆盖与代价。

只做结构层（不需要前向），因此 simplong 与 complong 都能跑。

单元定义：
  cell   : 只保留操作数所在单元格本身
  cross  : 以第一个操作数为锚，保留其所在行（全列）+ 所在列（全行）
  multi  : 对每个含操作数的表各取一个十字（锚点取该表内第一个操作数的行列），取并集
"""
import argparse
import json
import re

import numpy as np

from docmath_segs import parse_paragraphs

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
CONST = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 100.0, 365.0, 1000.0}


def nums(s):
    out = set()
    for m in NUM.findall(s or ""):
        t = m.replace(",", "").rstrip(".")
        try:
            out.add(round(abs(float(t)), 6))
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["simplong_testmini", "complong_testmini"])
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")

    for f in args.files:
        rows = [json.loads(l) for l in open(
            "D:/test_ollama/data/docmath/%s.jsonl" % f, encoding="utf-8")][:args.limit]
        cov = {"cell": [], "cross": [], "multi": [], "allx": []}
        cost = {"cell": [], "cross": [], "multi": [], "allx": []}
        ntab = []
        n = 0

        for r in rows:
            segs, diag = parse_paragraphs(r["paragraphs"])
            if not segs:
                continue
            want = {v for v in (nums(r["python_solution"]) - CONST)
                    if not (1900 <= v <= 2100)}
            cell_idx = [k for k, s in enumerate(segs)
                        if s["kind"] == "cell" and (nums(s["text"]) & want)]
            if len(cell_idx) < 2:
                continue
            cells = [segs[k] for k in cell_idx]

            # 表内行列 → 片段索引
            idx_of = {}
            for k, s in enumerate(segs):
                if s["kind"] in ("cell", "row_header") and s.get("tbl", -1) >= 0:
                    idx_of[(s["tbl"], s["row"], s["col"])] = k
            colname = {(s["tbl"], s["col"]): k for k, s in enumerate(segs)
                       if s["kind"] == "col_name"}

            def cross(tbl, row, col):
                out = set()
                for (t, rr, cc), k in idx_of.items():
                    if t == tbl and (rr == row or cc == col):
                        out.add(k)
                for (t, cc), k in colname.items():
                    if t == tbl and cc == col:
                        out.add(k)
                return out

            def toks_of(sids):
                return sum(len(tok(segs[k]["text"], add_special_tokens=False)["input_ids"])
                           for k in sids)

            L = sum(len(tok(s["text"], add_special_tokens=False)["input_ids"]) for s in segs)
            first = cells[0]
            u_cell = {cell_idx[0]}
            u_cross = cross(first["tbl"], first["row"], first["col"])
            u_multi = set()
            seen = {}
            for s in cells:
                seen.setdefault(s["tbl"], s)
            for t, s in seen.items():
                u_multi |= cross(t, s["row"], s["col"])

            def cover(unit):
                return sum(1 for k in cell_idx if k in unit) / len(cell_idx)

            cov["cell"].append(cover(u_cell))
            cov["cross"].append(cover(u_cross))
            cov["multi"].append(cover(u_multi))

            # 上界：每个操作数各取一个十字（真·闭包的近似）
            u_all = set()
            for s_ in cells:
                u_all |= cross(s_["tbl"], s_["row"], s_["col"])
            cov["allx"].append(cover(u_all))

            for name, unit in (("cell", u_cell), ("cross", u_cross),
                               ("multi", u_multi), ("allx", u_all)):
                cost[name].append(toks_of(unit) / max(1, L))
            ntab.append(len(seen))
            n += 1

        print("== %s（n=%d）" % (f, n))
        print("   含操作数的表数：中位 %.0f，最大 %d" % (np.median(ntab), max(ntab)))
        print("   %-7s %10s %12s %10s" % ("单元", "覆盖全部", "平均覆盖", "代价中位"))
        for k in ("cell", "cross", "multi", "allx"):
            v = np.array(cov[k])
            print("   %-7s %9.3f %12.3f %9.3f" % (
                k, (v >= 0.999).mean(), v.mean(), np.median(cost[k])))
        print()


if __name__ == "__main__":
    main()
