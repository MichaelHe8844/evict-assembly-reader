# -*- coding: utf-8 -*-
"""论元角色规则：从问句的线索短语判「谁是分子 / 谁是分母」。

不训练。目标只有一个：量清楚规则本身准不准，再决定要不要接进算式。
判据是 gold 式子给了方向（a-b / a/b / (a-b)/b 用 (i,j) 序，反之用 (j,i) 序）。
"""
import json
import re
import sys

sys.path.insert(0, ".")
from ground import tokens  # noqa: E402

NAMES = ["a+b", "a-b", "b-a", "a*b", "a/b", "b/a", "(a-b)/b", "(b-a)/a"]
DIR_IJ = {1, 4, 6}   # 用 (i,j) 序
DIR_JI = {2, 5, 7}   # 用 (j,i) 序


def close(a, b, t=1e-3):
    return a is not None and abs(a - b) <= t * max(1, abs(b))


def gold_dir(a, b, y):
    """1 = 用 (a,b) 序；-1 = 用 (b,a) 序；0 = 无方向（加/乘）或不可复现。"""
    for k, f in enumerate([
        lambda x, y_: x + y_, lambda x, y_: x - y_, lambda x, y_: y_ - x,
        lambda x, y_: x * y_, lambda x, y_: x / y_ if y_ else None,
        lambda x, y_: y_ / x if x else None,
        lambda x, y_: (x - y_) / y_ if y_ else None,
        lambda x, y_: (y_ - x) / x if x else None,
    ]):
        try:
            if close(f(a, b), y):
                return 1 if k in DIR_IJ else (-1 if k in DIR_JI else 0)
        except ZeroDivisionError:
            continue
    return 0


def toks(c):
    out = []
    for nm in (c.get("row", ""), c.get("col", "")):
        out += [t.lower() for t in tokens(nm) if len(t) >= 3]
    return out


def which_cell(span, cells):
    """span 里最先出现的单元格；找不到返回 None。"""
    low = span.lower()
    best, pos = None, 10 ** 9
    for idx, c in enumerate(cells):
        for t in toks(c):
            k = low.find(t)
            if k >= 0 and k < pos:
                best, pos = idx, k
    return best


# 每条规则：(名称, 正则, 处理函数) —— 返回 (分子下标, 分母下标) 或 None
PATTERNS = [
    ("percentage of", re.compile(r"(?:percentage|percent)\s+of", re.I)),
    ("ratio of X to Y", re.compile(r"ratio\s+of\s+(.{0,60}?)\s+to\s+(.{0,60})", re.I)),
    ("from X to Y", re.compile(r"from\s+(.{0,60}?)\s+to\s+(.{0,60})", re.I)),
]


def rule_dir(q, cells):
    """返回 (dir, rule_name)；dir: 1=(i,j) 序, -1=(j,i) 序, 0=未判。"""
    for name, rx in PATTERNS:
        m = rx.search(q)
        if not m:
            continue
        if name in ("ratio of X to Y", "from X to Y"):
            g1, g2 = which_cell(m.group(1), cells), which_cell(m.group(2), cells)
            if g1 is not None and g2 is not None and g1 != g2:
                # X to Y 与 from X to Y：前者 X 为分子，后者 Y 为分子
                num, den = (g1, g2) if name == "ratio of X to Y" else (g2, g1)
                return (1 if (num, den) == (0, 1) else -1), name
            continue
        # 百分比：分母是「总量」——名含 total 者优先，否则取数值较大者
        # （占比题里部分恒不大于总量，这条判据不依赖题面措辞）
        named = [k for k, c in enumerate(cells) if "total" in (
            (c.get("row", "") + " " + c.get("col", "")).lower())]
        if len(named) == 1:
            den = named[0]
        else:
            den = 0 if abs(cells[0]["v"]) >= abs(cells[1]["v"]) else 1
        num = 1 - den
        return (1 if (num, den) == (0, 1) else -1), name
    return 0, None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/r1_pairs_dev.json"
    rows = json.load(open(path, encoding="utf-8"))
    n = fired = 0
    hit = 0
    by_rule = {}
    conf = {}
    for r in rows:
        gp = [(i, j) for i, j, lab in r["pairs"] if lab]
        if not gp:
            continue
        i, j = gp[0]
        cells = [r["cands"][i], r["cands"][j]]
        a, b = cells[0]["v"], cells[1]["v"]
        gd = gold_dir(a, b, r["key"][1])
        if gd == 0:
            continue
        n += 1
        pd, rule = rule_dir(r["q"], cells)
        if pd != 0:
            fired += 1
            ok = (pd == gd)
            hit += ok
            d = by_rule.setdefault(rule, [0, 0])
            d[0] += 1
            d[1] += ok
        conf.setdefault((gd, pd), 0)
        conf[(gd, pd)] += 1
    print("有方向的题 n=%d；规则触发 %d (%.1f%%)" % (n, fired, 100 * fired / max(1, n)))
    print("规则在触发题上的方向准确率 %.3f (%d/%d)" % (hit / max(1, fired), hit, fired))
    print("规则在全部方向题上的覆盖×准确 = %.3f" % (hit / max(1, n)))
    print("\n分规则：")
    for k, (c, o) in sorted(by_rule.items(), key=lambda x: -x[1][0]):
        print("  %-20s 触发 %2d  准确 %.3f" % (k, c, o / max(1, c)))
    print("\n方向混淆（gold dir, 规则 dir）：", dict(conf))


if __name__ == "__main__":
    main()
