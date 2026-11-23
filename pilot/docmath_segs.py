# -*- coding: utf-8 -*-
"""
DocMath 段构造：把 paragraphs（含 markdown 表格）解析成与 FinQA 一致的片段序列。

片段 schema（与 common.build_dataset 对齐，供 table_index 直接使用）：
  kind ∈ {col_name, unit, row_header, cell, text, sep}
  text  原文
  row/col  表格内坐标（text 段为 -1）

约定：
  - 连续的 markdown 表格行合并为一张表；表与表之间 row 偏移，避免坐标串味
  - 第一行为表头 → col_name；表头里若含 "( in millions )" 之类的量纲，另出 unit 段
  - 数据行第一列 → row_header，其余 → cell
  - 其余段落 → text
"""
import re

UNIT_RE = re.compile(r"\(\s*(?:in\s+)?(thousand|million|billion|percent|share|unit|"
                     r"dollar|usd|eur|rmb)s?\b[^)]*\)", re.I)
NUM_RE = re.compile(r"^\(?[-+\$]?\s*\d[\d,]*(?:\.\d+)?\s*\)?%?$")


def _split_row(line):
    """'| a | b |' → ['a','b']；容忍缺失的首尾竖线。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip().replace("\u200b", "") for c in s.split("|")]


FILLER = {"$", "€", "£", "¥", "%", "(", ")", "-", "–", "—", "0", ""}


def _merge_fillers(cells):
    """
    DocMath 的表里，货币符号常被拆成独立单元格（'$' | '1,447'），
    导致数据行比表头宽，列对应关系整体错位。把纯符号格并入相邻数值格。
    """
    out = []
    i = 0
    while i < len(cells):
        c = cells[i]
        if c in ("$", "€", "£", "¥", "(", "-", "–", "—") and i + 1 < len(cells):
            nxt = cells[i + 1]
            if nxt and nxt not in FILLER:
                out.append((c + nxt).strip())
                i += 2
                continue
        out.append(c)
        i += 1
    return out


def _is_table_line(line):
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _is_sep_line(line):
    s = line.strip().strip("|")
    if not s:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in s.split("|") if c.strip())


def parse_paragraphs(paragraphs):
    """返回 (segs, tables_diag)。"""
    segs = []
    row_off = 0
    n_tables = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        if not _is_table_line(p.split("\n")[0] if "\n" in p else p):
            segs.append(dict(kind="text", text=p, row=-1, col=-1, tbl=-1))
            i += 1
            continue

        # 收集连续的表格段落（含段落内部的换行）
        block = []
        while i < len(paragraphs) and _is_table_line(
                paragraphs[i].split("\n")[0] if "\n" in paragraphs[i] else paragraphs[i]):
            block.extend([ln for ln in paragraphs[i].split("\n") if ln.strip()])
            i += 1

        rows = [_merge_fillers(_split_row(ln)) for ln in block if not _is_sep_line(ln)]
        if len(rows) < 2:
            segs.append(dict(kind="text", text="\n".join(block), row=-1, col=-1, tbl=-1))
            continue

        header, body = rows[0], rows[1:]
        n_col = max(len(r) for r in rows)
        n_tables += 1

        for j, h in enumerate(header):
            if not h:
                continue
            segs.append(dict(kind="col_name", text=h, row=row_off - 1, col=j, tbl=n_tables - 1))
            m = UNIT_RE.search(h)
            if m:
                segs.append(dict(kind="unit", text=m.group(0), row=row_off - 1, col=j, tbl=n_tables - 1))

        # 表头里单独一行的量纲说明（很多表把 "(in millions)" 放第一格）
        for j, h in enumerate(header):
            if j == 0 and UNIT_RE.search(h) and not any(
                    s["kind"] == "unit" and s["col"] == j for s in segs[-2 * n_col:]):
                segs.append(dict(kind="unit", text=UNIT_RE.search(h).group(0),
                                 row=row_off - 1, col=0, tbl=n_tables - 1))

        for r, cells in enumerate(body):
            rr = row_off + r
            for c, val in enumerate(cells):
                if val == "":
                    continue
                kind = "row_header" if c == 0 else "cell"
                segs.append(dict(kind=kind, text=val, row=rr, col=c, tbl=n_tables - 1))

        row_off += len(body)
    return segs, dict(n_tables=n_tables, n_seg=len(segs))


def doc_text(segs):
    return "\n".join(s["text"] for s in segs)


if __name__ == "__main__":
    import json
    path = "D:/test_ollama/data/docmath/complong_testmini.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:2]
    for r in rows:
        segs, diag = parse_paragraphs(r["paragraphs"])
        print("==", r["question_id"], diag)
        for s in segs[:12]:
            print("   [%-10s] r%-3d c%-2d %r" % (s["kind"], s["row"], s["col"], s["text"][:46]))
