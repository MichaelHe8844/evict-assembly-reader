# -*- coding: utf-8 -*-
"""
无监督锚点抽取：只给定「文档 + 问题」，不使用任何 gold 标注。

思路
----
1. 从线性化文档恢复表格结构 (row, col) —— 假定表格已被正确解析（parser 不在范围内）
2. 把问题切成内容词，与每个列名/行头做词面覆盖打分
3. 选出覆盖度达阈值的行与列；若一行/列都没命中，则退化为「选中列的全部行」
4. 锚点组 = 选中单元格 + 其行头 + 其列名 + 单位
   若 `cross=True`，再补上「匹配到的列」的整列（连同各行行头），得到算术任务所需
   的**十字**单元（见 arm_c.slice_anchor 与论文 3.1 节）。只扩匹配到的列，
   不能用对冲扩出来的「全部列」去扩，否则等价于保留整表。

刻意保持简单：这是论文里该被超越的 baseline，不是最终方法。
"""
import re
from typing import List, Dict, Tuple

from common import UNIT_TEXT_RE

STOP = set("""a an the of for and or in on to is are was were be been being what which who
how many much percent fiscal ended december january february march april may june july august
september october november does did do with by from at as that this these those it its per
company companies inc ltd corp corporation group reported report please question answer""".split())

UNIT_HINT = re.compile(r"\(\s*(thousand|million|billion|percent|share|unit)", re.I)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def years(s: str):
    return set(YEAR_RE.findall(s or ""))


def norm_word(w: str) -> str:
    w = w.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def words(s: str):
    return {norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", s or "")
            if norm_word(w) not in STOP and len(w) > 1}


def coverage(qw, header: str) -> float:
    """header 的内容词有多大比例出现在问题里。"""
    hw = words(header)
    if not hw:
        return 0.0
    return len(qw & hw) / len(hw)


def table_index(segs) -> Tuple[Dict, Dict, Dict, int, int]:
    """从片段元数据恢复表格坐标。"""
    col_name, unit, row_head, cell = {}, {}, {}, {}
    n_col = n_row = 0
    for i, s in enumerate(segs):
        if s["kind"] == "col_name":
            col_name[s["col"]] = i
            n_col = max(n_col, s["col"] + 1)
        elif s["kind"] == "unit":
            unit[s["col"]] = i
        elif s["kind"] == "row_header":
            row_head[s["row"]] = i
            n_row = max(n_row, s["row"] + 1)
        elif s["kind"] == "cell":
            cell[(s["row"], s["col"])] = i
            n_row = max(n_row, s["row"] + 1)
            n_col = max(n_col, s["col"] + 1)
    return col_name, unit, row_head, cell, n_row, n_col


def extract(segs, question: str, thresh: float = 0.6, max_cells=None,
            fuzzy: bool = True, use_year: bool = True, hedge: bool = True,
            cross: bool = False, max_cross_cols: int = 2, cand2: bool = True,
            year_cols: bool = False, emb=None, emb_w: float = 1.0,
            use_lex: bool = True):
    """
    返回 (选中片段索引列表, 诊断信息)。

    cross=True 时把「匹配到的列」扩成整列，得到算术任务所需的十字单元。
    max_cross_cols 默认 2：实测 1 与 2 的操作数覆盖率相同（0.875），
    再往上只增加代价不增加覆盖，因为剩余缺口来自行列选错而非扩得不够。
    """
    qw = words(question)
    q_norm = re.sub(r"\s+", " ", question.lower()).strip()
    q_years = years(question)
    col_name, unit, row_head, cell, n_row, n_col = table_index(segs)

    def score(header: str) -> float:
        """词面（可选）+ 向量（可选）的联合打分。

        emb 是一个把表头映射到 [0,1] 相似度的可调用对象（见 emb_extract.py）。
        默认 None 时与原文逐位相同，不影响既有结果。
        """
        hw = words(header)
        h = re.sub(r"\s+", " ", (header or "").lower()).strip()
        s = 0.0
        if use_lex and hw:
            exact = len(qw & hw) / len(hw)
            s = exact
            if fuzzy and exact < 1.0:
                # 前缀 5 字符：处理 sales/sale、purchased/purchase、assets/asset
                m = sum(1 for w in hw if any(w[:5] == x[:5] for x in qw))
                s = max(s, 0.85 * m / len(hw))
        if use_lex and use_year and s < 0.6:
            hy = years(header)
            if hy & q_years:          # 日期型表头：年份是最强信号
                s = max(s, 0.70)
        if use_lex and s == 0.0 and h and h in q_norm:
            s = 1.0
        if emb is not None and h:
            s = max(s, emb_w * float(emb(header)))
        return s

    cscore = {j: score(segs[i]["text"]) for j, i in col_name.items()}
    rscore = {r: score(segs[i]["text"]) for r, i in row_head.items()}

    cols = [j for j, s in cscore.items() if s >= thresh]
    rows = [i for i, s in rscore.items() if s >= thresh]
    # 十字扩展只能扩「真正匹配到的列」。若用对冲后的 cols（=全部列）去扩，
    # 每个列都扩成整列，等价于保留整张表，预算直接爆掉。
    cols_matched = list(cols)

    fallback_col = False
    # 关键的对冲：行头可信（只命中 1–2 行）而列有歧义时，保留该行的**全部列**。
    # 只涉及 1–2 行 × n_cols 个单元格，代价小，却能把「列选错」这一最大失败源消掉。
    row_confident = 0 < len(rows) <= 2
    col_ambiguous = (not cols) or (max(cscore.values() or [0]) < 0.8)
    if hedge and row_confident and col_ambiguous:
        cols = sorted(col_name.keys())
        fallback_col = True
    elif not cols and cscore:
        best = max(cscore.values())
        cols = [j for j, s in cscore.items() if s == best] if best > 0 else \
               [max(cscore, key=cscore.get)]
        fallback_col = True
    fallback_row = False
    if not rows:
        # 没有行命中：不猜行，保留选中列的所有行
        rows = sorted(row_head.keys())
        fallback_row = True

    # 亚军候选：金标行列有 26%/21% 排在第 2 名，却因够不到阈值被丢掉。
    # 补上分数 ≥0.3 的前两名，代价仅 +0.5% 文档长度，答案格命中 87.0%→90.3%。
    if cand2:
        extra_r = {k for k, s in sorted(rscore.items(), key=lambda kv: -kv[1])[:2]
                   if s >= 0.3}
        extra_c = {k for k, s in sorted(cscore.items(), key=lambda kv: -kv[1])[:2]
                   if s >= 0.3}
        rows = sorted(set(rows) | extra_r)
        cols = sorted(set(cols) | extra_c)

    # 年份定向补列：「increase between 2016 and 2017」这类问题需要两列都在。
    # 默认关闭——它 +2.4pp 答案格命中，但代价从 7.9% 涨到 10.1%，
    # 是否值得要端到端确认（干扰数增加会抵消覆盖收益）。
    if year_cols:
        qy = years(question)
        ycols = sorted({j for j, i in col_name.items()
                        if years(segs[i]["text"]) & qy})
        if qy and 0 < len(ycols) <= 3:
            cols = sorted(set(cols) | set(ycols))

    # 若锚点集超出预算，按 (行分+列分) 排序裁剪单元格；表头与单位始终保留
    pairs = [(i, j) for i in rows for j in cols if (i, j) in cell]
    if max_cells is not None and len(pairs) > max_cells:
        pairs.sort(key=lambda ij: -(rscore.get(ij[0], 0) + cscore.get(ij[1], 0)))
        pairs = pairs[:max_cells]

    idx = set()
    for (i, j) in pairs:
        idx.add(cell[(i, j)])
        if i in row_head:
            idx.add(row_head[i])
        if j in col_name:
            idx.add(col_name[j])
        if j in unit:
            idx.add(unit[j])

    # 十字扩展：单步算术的第二个操作数常在**同一列的另一行**，
    # 只保留选中行会漏掉它。补上「匹配到的列」的整列（含各行行头）。
    if cross and cols_matched:
        for j in sorted(cols_matched, key=lambda k: -cscore.get(k, 0))[:max_cross_cols]:
            if j in col_name:
                idx.add(col_name[j])
            if j in unit:
                idx.add(unit[j])
            for i in row_head:
                if (i, j) in cell:
                    idx.add(cell[(i, j)])
                    idx.add(row_head[i])

    # 单位句（正文里的 "amounts in millions"）总是保留，代价极小
    for i, s in enumerate(segs):
        if s["kind"] == "text" and (UNIT_TEXT_RE.search(s["text"]) or UNIT_HINT.search(s["text"])):
            idx.add(i)
            break

    diag = dict(cols=cols, cols_matched=cols_matched, rows=rows, n_pairs=len(pairs),
                cross=cross, fallback_col=fallback_col, fallback_row=fallback_row,
                cscore=cscore, rscore=rscore)
    return sorted(idx), diag


def probe_question(segs, gold_group):
    """复刻 arm_c 的抽取探针问题。"""
    ci = gold_group["C"][0] if gold_group.get("C") else None
    ri = gold_group["R"][0] if gold_group.get("R") else None
    if ci is None or ri is None:
        return None
    return "What is the value of %s for %s?" % (segs[ci]["text"], segs[ri]["text"])
