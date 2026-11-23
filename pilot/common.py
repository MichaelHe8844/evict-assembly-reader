# -*- coding: utf-8 -*-
"""
公共模块：把 FinQA 文档构造成「结构保留的行式线性化上下文」，
并把 gold 数值单元格扩展成「数值锚点组」。

锚点组 (anchor group) 定义 —— 一个数值要能被正确复算，至少需要四个成分共存：
  V  value cell      : 数值本身
  R  row header      : 该数值属于哪个主体/项目行
  C  column name     : 该数值是哪一列（指标名）
  U  unit            : 单位/量纲（thousands / millions / billions / %）
缺任一项，数值就退化为「无主体的裸数」，这正是金融数值推理里
期间错配、主体错配、量级错配的来源。
"""
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

UNIT_RE = re.compile(r"\(\s*(thousands|millions|billions|percent|shares|units)\s*\)", re.I)
# 正文中的量纲声明，如 "amounts in millions" / "dollars in thousands"
UNIT_TEXT_RE = re.compile(
    r"\b(?:amounts?\s+)?(?:in|are\s+in|expressed\s+in)\s+(thousands|millions|billions)\b",
    re.I)
NUM_RE = re.compile(r"^[\-\+]?[\d,]*\.?\d+$")
PUNCT_ONLY = re.compile(r"^[\s\.\,\;\:\-\|\%\(\)\[\]]*$")


def norm_num(s: str) -> Optional[str]:
    """把 '$ 2,457' / '( 637 )' 归一成可比较的数字串。"""
    if s is None:
        return None
    t = str(s).strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    if not NUM_RE.match(t):
        return None
    if neg and not t.startswith("-"):
        t = "-" + t
    return t


@dataclass
class Seg:
    """上下文中的一个片段，带类型与表格坐标。"""
    sid: str
    text: str
    kind: str                 # text | col_name | unit | col_header | row_header | cell | sep
    row: int = -1
    col: int = -1
    tok_start: int = -1       # 由 tokenizer 回填
    tok_end: int = -1


def split_col_header(h: str):
    """把 'payments volume ( billions )' 拆成 ('payments volume', '( billions )')。"""
    m = UNIT_RE.search(h)
    if not m:
        return h, None
    return h[:m.start()].strip(), m.group(0)


def build_document(example: dict) -> List[Seg]:
    """
    构造上下文片段序列。表格按「行式」线性化：
        col0 | col1 | col2 ...
        r1c0 | r1c1 | ...
    表头只出现一次 —— 这是真实 CSV/10-K 表格的形态，也是单位成为
    单点故障(single point of failure)的原因。
    """
    segs: List[Seg] = []

    def add(text, kind, row=-1, col=-1):
        if text is None:
            return
        text = str(text).strip()
        if not text:
            return
        # 注意：分隔符本身就是纯标点，不能过滤 —— 否则单元格会粘连成
        # "operating income1150310815" 这样的串，数字边界消失。
        if kind != "sep" and PUNCT_ONLY.match(text):
            return
        segs.append(Seg(sid=f"s{len(segs)}", text=text, kind=kind, row=row, col=col))

    # 正文（前）
    for p in example.get("pre_text", []) or []:
        add(p, "text")
    add("\nTABLE:\n", "sep")

    table = example.get("table", [])
    n_row = len(table)
    n_col = len(table[0]) if n_row else 0

    # 表头行：列名 / 单位拆成独立片段
    for j in range(n_col):
        raw = table[0][j]
        if j == 0:
            add(raw, "col_header", 0, 0)
        else:
            name, unit = split_col_header(raw)
            add(name, "col_name", 0, j)
            if unit:
                add(unit, "unit", 0, j)
        add("|", "sep")

    # 数据行
    for i in range(1, n_row):
        for j in range(n_col):
            kind = "row_header" if j == 0 else "cell"
            add(table[i][j], kind, i, j)
            add("|", "sep")
        add("\n", "sep")

    add("\nTEXT:\n", "sep")
    for p in example.get("post_text", []) or []:
        add(p, "text")

    return segs


def find_gold_cells(example: dict):
    """
    用 qa.steps 里的数值实参 + ann_table_rows 定位 gold 单元格。
    返回 [(row, col, value_str), ...]
    """
    qa = example.get("qa", {})
    steps = qa.get("steps", []) or []
    rows = qa.get("ann_table_rows", []) or []
    table = example.get("table", [])

    wanted = set()
    for st in steps:
        for k in ("arg1", "arg2"):
            v = norm_num(st.get(k))
            if v is not None:
                wanted.add(v)

    hits = []
    for i in rows:
        if i <= 0 or i >= len(table):
            continue
        for j in range(1, len(table[i])):
            v = norm_num(table[i][j])
            if v is not None and v in wanted:
                hits.append((i, j, str(table[i][j]).strip()))
    return hits


def find_unit_text(segs: List[Seg]) -> Optional[int]:
    """正文里承载量纲声明的句子（'amounts in millions'）。"""
    for idx, s in enumerate(segs):
        if s.kind == "text" and UNIT_TEXT_RE.search(s.text):
            return idx
    return None


def anchor_group(example: dict, segs: List[Seg], row: int, col: int):
    """
    返回 (group, unit_source)。
    四个成分：V 数值 / R 行头 / C 列名 / U 单位。
    单位优先取表头括号；表头没有则回退到正文的量纲声明句 ——
    后者正是「长距离绑定」，也是 FinQA 里的常态。
    """
    g = {"V": [], "R": [], "C": [], "U": []}
    for idx, s in enumerate(segs):
        if s.kind == "cell" and s.row == row and s.col == col:
            g["V"].append(idx)
        elif s.kind == "row_header" and s.row == row:
            g["R"].append(idx)
        elif s.kind == "col_name" and s.col == col:
            g["C"].append(idx)
        elif s.kind == "unit" and s.col == col:
            g["U"].append(idx)

    if g["U"]:
        return g, "header"
    ut = find_unit_text(segs)
    if ut is not None:
        g["U"] = [ut]
        return g, "text"
    return g, None


def build_dataset(path: str):
    """产出可直接送入分析的结构化样本。"""
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for ex in data:
        table = ex.get("table", [])
        if len(table) < 3 or len(table[0]) < 3:
            continue
        hits = find_gold_cells(ex)
        if not hits:
            continue
        segs = build_document(ex)
        for (r, c, val) in hits:
            g, usrc = anchor_group(ex, segs, r, c)
            if not g["V"] or not g["R"]:
                continue
            out.append({
                "id": ex.get("id"),
                "filename": ex.get("filename"),
                "question": ex["qa"].get("question"),
                "exe_ans": ex["qa"].get("exe_ans"),
                "gold_value": val,
                "row": r, "col": c,
                "unit_source": usrc,
                "has_unit": len(g["U"]) > 0,
                "has_col": len(g["C"]) > 0,
                "segs": [asdict(s) for s in segs],
                "group": g,
            })
    return out


if __name__ == "__main__":
    import sys
    from collections import Counter
    ds = build_dataset(sys.argv[1])
    print("samples:", len(ds))
    c = Counter(d["unit_source"] for d in ds)
    print("unit_source:", dict(c))
    print("with col name: %d" % sum(d["has_col"] for d in ds))
    d = ds[0]
    for s in d["segs"][:30]:
        print(f'  [{s["kind"]:>10}] r{s["row"]:>2} c{s["col"]:>2}  {s["text"][:68]}')
    print("group:", d["group"], "unit_source:", d["unit_source"])
