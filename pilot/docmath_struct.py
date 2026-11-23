# -*- coding: utf-8 -*-
"""
DocMath 结构层分析（不需要前向，因此 3 万 token 的 complong 也能做）。

回答三个问题：
  1. 长文档里绑定距离是否真的放大（对比 FinQA 的 median 114）
  2. 一次算式的操作数是否还在同一张表里（跨表则单个「十字」不够）
  3. 单元格单元 vs 十字单元对操作数的覆盖，在长文档上如何变化
"""
import json
import re
import argparse
import numpy as np

from docmath_segs import parse_paragraphs

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


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
    ap.add_argument("--files", nargs="+",
                    default=["simplong_testmini", "complong_testmini"])
    ap.add_argument("--limit", type=int, default=0, help="0 表示不限制")
    ap.add_argument("--tokenizer", default="E:/KV_models/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for f in args.files:
        path = "D:/test_ollama/data/docmath/%s.jsonl" % f
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if args.limit:
            rows = rows[:args.limit]

        Ls, dists, span_same_tbl = [], [], []
        cell_cov, cross_cov = [], []
        n_operand = []

        for r in rows:
            segs, diag = parse_paragraphs(r["paragraphs"])
            if not segs:
                continue
            # 分词与位置
            pos, off = [], 0
            for s in segs:
                n = len(tok(s["text"], add_special_tokens=False)["input_ids"])
                pos.append((off, off + n))
                off += n
            L = off
            if L < 64:
                continue
            Ls.append(L)

            # 操作数：解答里出现的数字，且能在表格单元格里找到
            CONST = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 100.0, 365.0, 1000.0}
            # 年份不是操作数：2018 / 2017 会命中表内「第二行表头」，把样本污染成
            # 「62 个单元格、27 张表」这种荒谬结果。1900–2100 全部剔除。
            want = {v for v in (nums(r["python_solution"]) - CONST)
                    if not (1900 <= v <= 2100)}
            cells = [(i, s) for i, s in enumerate(segs) if s["kind"] == "cell"
                     and (nums(s["text"]) & want)]
            if len(cells) < 2:
                continue
            n_operand.append(len(cells))

            # 1) 绑定距离：每个操作数格到其行头/列名的最大跨度
            head = {}
            for i, s in enumerate(segs):
                if s["kind"] == "row_header":
                    head[("r", s["row"])] = i
                elif s["kind"] == "col_name":
                    head[("c", s["col"])] = i
            for i, s in cells:
                idx = [i]
                if ("r", s["row"]) in head:
                    idx.append(head[("r", s["row"])])
                if ("c", s["col"]) in head:
                    idx.append(head[("c", s["col"])])
                a = min(pos[j][0] for j in idx)
                b = max(pos[j][1] for j in idx)
                dists.append(b - a)

            # 2) 操作数是否同表：用 (row, col) 所在表格的行偏移区间近似
            span_same_tbl.append(len({s.get("tbl", -1) for _, s in cells}) == 1)

            # 3) 覆盖：单元格单元 = 只保操作数格 + 各自行头列名；
            #    十字单元 = 第一个操作数所在行全列 + 所在列全行
            first = cells[0][1]
            same_row = sum(1 for _, s in cells if s["row"] == first["row"])
            same_col = sum(1 for _, s in cells if s["col"] == first["col"])
            cross_cov.append((same_row + same_col - 1) / len(cells))

        Ls = np.array(Ls)
        dists = np.array(dists) if dists else np.array([0])
        print("== %s（n=%d 有效）" % (f, len(Ls)))
        print("   token 中位 %.0f  p90 %.0f  max %.0f" % (
            np.median(Ls), np.percentile(Ls, 90), Ls.max()))
        print("   绑定距离 中位 %.0f  p90 %.0f  max %.0f" % (
            np.median(dists), np.percentile(dists, 90), dists.max()))
        print("   每个问题的操作数个数 中位 %.0f" % np.median(n_operand))
        print("   操作数全在同一张表的样本：%.3f" % np.mean(span_same_tbl))
        print("   单个十字覆盖全部操作数的平均比例：%.3f" % np.mean(cross_cov))
        print()


if __name__ == "__main__":
    main()
