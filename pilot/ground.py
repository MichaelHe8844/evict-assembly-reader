# -*- coding: utf-8 -*-
"""问句→单元格定位模块（Q2C-Grounder）。

只依赖「问题 + 文档」，不需要 gold：
  1. 把表格展开成单元格索引（行头、列名、取值）；
  2. 词面 + 年份 + 度量类型三路打分；
  3. 输出 top-k 候选，并可抽出「两个不同行」的一对（算术题通常要两个操作数）。

设计动机来自 5.13 的分解：错误的主要形态不是「数字被删」，而是
「只锁定了两个操作数中的一个」。所以模块的第一职责是**同时给出多个候选**，
而不是把单一候选排得更准。
"""
import math
import re
from collections import Counter

STOP = set("""a an the of for in on at to from by and or is are was were be been being
what which who whom whose how much many does did do done has have had with as that this
these those it its their company companies year years value values amount total
please answer question data table reported report""".split())

TOK = re.compile(r"[a-z0-9][a-z0-9\.\-/%]*")
NUM = re.compile(r"-?\d[\d,]*\.?\d*%?")


def to_float(tok):
    t = str(tok).strip()
    pct = t.endswith("%")
    t = t.rstrip("%").replace(",", "").replace("$", "").strip()
    try:
        v = float(t)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def _norm(t):
    """粗归一：单复数、所有格。FinQA 的问句与表头经常只差一个 s（transaction/transactions）。"""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokens(s):
    return [_norm(t) for t in TOK.findall(str(s).lower()) if t not in STOP and len(t) > 1]


def parse_table(entry):
    """返回 (cells, headers)。cells 里每项含 r/c/row/col/v/raw。"""
    t = entry.get("table") or []
    if not t:
        return [], []
    cols = [str(x).strip() for x in t[0]]
    cells = []
    for r in range(1, len(t)):
        row = t[r]
        rh = str(row[0]).strip()
        for c in range(1, len(row)):
            v = to_float(row[c])
            if v is None:
                continue
            cells.append(dict(r=r, c=c, v=v, row=rh, row_t=tokens(rh),
                              col=cols[c] if c < len(cols) else "",
                              col_t=tokens(cols[c]) if c < len(cols) else [],
                              raw=str(row[c]).strip()))
    return cells, cols


def build_idf(cells):
    df = Counter()
    for c in cells:
        for t in set(c["row_t"]) | set(c["col_t"]):
            df[t] += 1
    n = max(1, len(cells))
    return {t: math.log(1 + n / (1 + k)) for t, k in df.items()}


# 度量类型提示：问题问什么，就优先哪一列
METRIC_HINTS = [
    (re.compile(r"percent|percentage|rate|ratio|margin|growth"), re.compile(r"%|percent")),
    (re.compile(r"average|per\b|mean"), re.compile(r"average|per ")),
    (re.compile(r"total|aggregate|combined"), re.compile(r"total")),
    (re.compile(r"number of|count|units"), re.compile(r"number|units|count")),
]


def score_cells(entry, question, topk=6):
    cells, cols = parse_table(entry)
    if not cells:
        return []
    idf = build_idf(cells)
    q = tokens(question)
    qset = set(q)
    qnums = {round(v, 6) for v in (to_float(m) for m in NUM.findall(question)) if v is not None}

    mh = [pat for rx, pat in METRIC_HINTS if rx.search(question.lower())]

    out = []
    for c in cells:
        rs = sum(idf.get(t, 1.0) for t in c["row_t"] if t in qset)
        cs = sum(idf.get(t, 1.0) for t in c["col_t"] if t in qset)
        rs /= math.sqrt(max(1, len(c["row_t"])))
        cs /= math.sqrt(max(1, len(c["col_t"])))
        s = 1.0 * rs + 1.2 * cs
        # 年份/编号对齐：问题里出现的数字若与列名一致，强信号
        for n in qnums:
            if n == int(n) and re.search(r"\b%d\b" % int(n), c["col"]):
                s += 1.5
            if n == int(n) and re.search(r"\b%d\b" % int(n), c["row"]):
                s += 0.8
        # 度量类型一致
        for pat in mh:
            if pat.search(c["col"].lower()):
                s += 0.6
        out.append((s, c))
    out.sort(key=lambda x: -x[0])
    return out[:topk]


def pick_pair(ranked, B_cells=2):
    """从候选里挑一对「不同行」的格：算术题的两个操作数通常分行。"""
    if not ranked:
        return []
    if len(ranked) == 1:
        return [ranked[0]]
    best = (ranked[0], ranked[1], -1e9)
    for i in range(min(len(ranked), 6)):
        for j in range(i + 1, min(len(ranked), 6)):
            a, b = ranked[i][1], ranked[j][1]
            bonus = 0.0
            if a["r"] != b["r"]:
                bonus += 0.8
            if a["c"] != b["c"]:
                bonus += 0.5
            s = ranked[i][0] + ranked[j][0] + bonus
            if s > best[2]:
                best = (ranked[i], ranked[j], s)
    return [best[0], best[1]]


# ---------------- 与实验脚本对接：按 seg 结构建索引 ----------------

def cells_from_segs(segs, spans, keep_tokens):
    """只保留「整格 token 都在保留集内、且行头/列名也在」的单元格。

    这样做有两个作用：
      1. 模块看不到被压缩掉的格，避免把证据偷偷塞回上下文；
      2. 模型引用 C# 后能按单元格值展开，展开的值一定真的在上下文里。
    """
    rh, cn, un = {}, {}, {}
    for i, s in enumerate(segs):
        a, b = spans[i]
        if a < 0:
            continue
        k = s.get("kind")
        if k == "row_header":
            rh[s.get("row")] = s["text"]
        elif k in ("col_name", "col_header"):
            cn[s.get("col")] = s["text"]
        elif k == "unit":
            un[s.get("col")] = s["text"]

    def alive(i):
        a, b = spans[i]
        return a >= 0 and all(t in keep_tokens for t in range(a, b))

    cells = []
    for i, s in enumerate(segs):
        if s.get("kind") != "cell":
            continue
        a, b = spans[i]
        if a < 0 or not alive(i):
            continue
        v = to_float(s["text"])
        if v is None:
            continue
        r, c = s.get("row"), s.get("col")
        cells.append(dict(r=r, c=c, v=v, raw=s["text"].strip(),
                          row=rh.get(r, ""), col=cn.get(c, ""), unit=un.get(c, ""),
                          row_t=tokens(rh.get(r, "")), col_t=tokens(cn.get(c, ""))))
    return cells


def rank(cells, question, topk=6):
    """与 score_cells 同一套打分，只是输入换成已建好的索引。"""
    if not cells:
        return []
    idf = build_idf(cells)
    qset = set(tokens(question))
    qnums = {round(v, 6) for v in (to_float(m) for m in NUM.findall(question)) if v is not None}
    mh = [pat for rx, pat in METRIC_HINTS if rx.search(question.lower())]
    out = []
    for c in cells:
        rs = sum(idf.get(t, 1.0) for t in c["row_t"] if t in qset) / math.sqrt(max(1, len(c["row_t"])))
        cs = sum(idf.get(t, 1.0) for t in c["col_t"] if t in qset) / math.sqrt(max(1, len(c["col_t"])))
        s = rs + 1.2 * cs
        for n in qnums:
            if n == int(n) and re.search(r"\b%d\b" % int(n), c["col"]):
                s += 1.5
            if n == int(n) and re.search(r"\b%d\b" % int(n), c["row"]):
                s += 0.8
        for pat in mh:
            if pat.search(c["col"].lower()):
                s += 0.6
        out.append((s, c))
    out.sort(key=lambda x: -x[0])
    return out[:topk]


def render_block(ranked, with_value=True):
    """候选块。

    with_value=True 时带上数值 —— 这些格**确实在保留集里**（cells_from_segs 已过滤），
    所以不是把证据偷偷塞回去，只是把分散在线性化表格里的同一批 token 重新排布一次。
    """
    lines = []
    for k, (_, c) in enumerate(ranked, 1):
        unit = (" " + c["unit"]).strip() if c.get("unit") else ""
        v = (" = %s" % c["raw"]) if with_value else ""
        lines.append("C%d: %s | %s%s%s" % (k, c["row"], c["col"], unit, v))
    return "\n".join(lines)


def expand_cids(text, ranked):
    """把 C1/C2 换成单元格数值；无法解析的 id 原样留着（会在执行时报错）。"""
    vals = {("C%d" % k): c["v"] for k, (_, c) in enumerate(ranked, 1)}

    def fmt(v):
        # 不能用 %g：2,157,503 会被写成 2.1575e+06，safe_eval 会被截成 2.1575（踩过）
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return ("%.6f" % v).rstrip("0").rstrip(".")

    def sub(m):
        k = m.group(0)
        v = vals.get(k)
        if v is None:
            return k
        return fmt(v)

    return re.sub(r"\bC\d+\b", sub, text)


def used_cids(text, ranked):
    ok = {"C%d" % k for k in range(1, len(ranked) + 1)}
    return sorted({m.group(0) for m in re.finditer(r"\bC\d+\b", text or "") if m.group(0) in ok},
                  key=lambda s: int(s[1:]))


def parse_selection(text):
    """解析「Cells: C1, C3 / Expression: C1 - C3」两行输出。

    返回 (引用的 ID 列表, 算式文本)。模型不必复述数值——数值由解释器按 ID 取。
    """
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cids, expr = [], ""
    for ln in lines:
        low = ln.lower()
        if low.startswith("cells") or low.startswith("cell"):
            body = ln.split(":", 1)[1] if ":" in ln else ln
            cids = re.findall(r"\bC\d+\b", body)
        elif "expression" in low:
            expr = ln.split(":", 1)[1].strip() if ":" in ln else ln
    if not expr:
        # 没有 Expression 行：取最后一行里含运算符的
        for ln in reversed(lines):
            if re.search(r"[+\-*/()]", ln) and not ln.lower().startswith("cells"):
                expr = ln.split(":", 1)[1].strip() if ":" in ln else ln
                break
    if not cids:
        cids = re.findall(r"\bC\d+\b", expr or text)
    cids = list(dict.fromkeys(cids))   # 保序去重
    return cids, expr
