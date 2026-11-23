# -*- coding: utf-8 -*-
"""把"选错单元格"拆成两种机制，并定位错在哪一行/哪一列。

两类必须分开：
  (1) 全上下文也对不了  → 定位能力不足，与压缩无关
  (2) 全上下文对、压缩后错、且 gold 格仍在 → 压缩使定位漂移（本文能改的部分）
  (3) 全上下文对、压缩后错、gold 格已丢 → 覆盖率问题（已由 5.7/5.11 处理）

同时统计错位方向（行错 / 列错 / 行列都错），这决定模块该做哪一半。
"""
import json
import re
from collections import Counter

NUM = re.compile(r"-?\d[\d,]*\.?\d*%?")
SINGLE = re.compile(r"^([a-z]+)\(([^()]*)\)$")


def to_float(tok):
    t = str(tok).strip()
    pct = t.endswith("%")
    t = t.rstrip("%").replace(",", "").replace("$", "").strip()
    try:
        v = float(t)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def numbers(s):
    out = []
    for m in NUM.findall(s or ""):
        v = to_float(m)
        if v is not None:
            out.append(v)
    return out


def close(a, b, rel=0.005):
    return a is not None and b is not None and abs(a - b) <= rel * max(1e-9, abs(b))


def grid(entry):
    """把 table 展开成单元格清单：行头（首列）、列名（首行）、取值。"""
    t = entry.get("table") or []
    if not t:
        return []
    cols = [str(x).strip().lower() for x in t[0]]
    out = []
    for r in range(1, len(t)):
        row = t[r]
        rh = str(row[0]).strip().lower() if row else ""
        for c in range(1, len(row)):
            v = to_float(row[c])
            if v is None:
                continue
            out.append(dict(r=r, c=c, v=v, row=rh,
                            col=cols[c] if c < len(cols) else "",
                            raw=str(row[c])))
    return out


def gold_args(entry):
    """返回 (op, 表格实参列表)。const_x 是程序常量（如 const_100），不是单元格。"""
    prog = ((entry.get("qa") or {}).get("program_re") or "").strip()
    m = SINGLE.match(prog)
    if not m:
        return None
    args = []
    for a in m.group(2).split(","):
        a = a.strip()
        if a.startswith("const_"):
            continue
        v = to_float(a)
        if v is None:
            return None
        args.append(v)
    if not args:
        return None
    return m.group(1), args


def gold_cells(entry):
    """用 gold_inds 标注的行 + 程序实参，反查 gold 单元格 (r, c)。"""
    g = gold_args(entry)
    if not g:
        return None
    _, args = g
    rows = set()
    for k in (entry["qa"].get("gold_inds") or {}):
        if k.startswith("table_"):
            try:
                rows.add(int(k.split("_")[1]))
            except ValueError:
                pass
    if not rows:
        # 没标注就在全表找
        rows = set(range(len(entry.get("table") or [])))
    gs = grid(entry)
    out = []
    for a in args:
        hit = [c for c in gs if c["r"] in rows and close(c["v"], a, 1e-3)]
        if not hit:
            hit = [c for c in gs if close(c["v"], a, 1e-3)]
        if hit:
            out.append(hit[0])
        else:
            out.append(None)
    return out


def model_cells(rec, entry):
    """模型写出的数字，各自落在哪个格子上（不落格则为 None）。"""
    gs = grid(entry)
    out = []
    for v in numbers(rec.get("text", "")):
        if v in (1.0, 100.0, 1000.0, 0.01, 0.1, 0.001):
            continue
        hit = [c for c in gs if close(c["v"], v, 1e-3)]
        out.append((v, hit[0] if hit else None))
    return out


def offset(gold, used):
    """错位方向：row=同行换列，col=同列换行，both=都变。"""
    if gold["r"] == used["r"] and gold["c"] != used["c"]:
        return "列错位"
    if gold["c"] == used["c"] and gold["r"] != used["r"]:
        return "行错位"
    return "行列都错"


d = json.load(open("../results/cand2_confirm_rescored.json", encoding="utf-8"))
raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
by = {}
for r in d["records"]:
    by.setdefault((r["id"], round(float(r["gold"]), 6)), {})[r["cond"]] = r
pairs = [(k, c["full"], c["auto_f@20"]) for k, c in by.items()
         if "full" in c and "auto_f@20" in c]

buckets = Counter()
offsets = {"case1": Counter(), "case2": Counter()}
examples = {"case1": [], "case2": []}

for k, f, c in pairs:
    entry = raw.get(k[0], {})
    gc = gold_cells(entry)
    if not gc or any(x is None for x in gc):
        buckets["gold 格解析失败"] += 1
        continue
    gold = gc[0]
    used = model_cells(c, entry)
    table_hits = [(v, x) for v, x in used if x is not None]
    if not table_hits:
        continue                      # 不是"取了表里另一个格"，归入别类
    # 是否真的没用到 gold 值
    if any(close(v, gold["v"], 1e-3) for v, _ in used):
        continue                      # 用到了 gold 格，属算错而非选错
    keep_gold = bool(c.get("num")) or (c.get("air") or 0) > 0.99
    if f["correct"] and not c["correct"]:
        tag = "case2" if keep_gold else "coverage"
        if tag == "case2":
            off = offset(gold, table_hits[0][1])
            offsets["case2"][off] += 1
            if len(examples["case2"]) < 8:
                examples["case2"].append(
                    (k[0][:24], gold["row"], gold["col"], gold["v"],
                     table_hits[0][1]["row"], table_hits[0][1]["col"], table_hits[0][0], off))
        buckets[tag] += 1
    elif not f["correct"] and not c["correct"]:
        off = offset(gold, table_hits[0][1])
        offsets["case1"][off] += 1
        buckets["case1"] += 1
        if len(examples["case1"]) < 8:
            examples["case1"].append(
                (k[0][:24], gold["row"], gold["col"], gold["v"],
                 table_hits[0][1]["row"], table_hits[0][1]["col"], table_hits[0][0], off))

print("配对 %d 题" % len(pairs))
for k, v in buckets.most_common():
    print("  %-16s %d" % (k, v))
print("\n错位方向：")
for name in ("case1", "case2"):
    tot = sum(offsets[name].values())
    print("  %s (n=%d): %s" % (name, tot, dict(offsets[name]) if tot else "-"))
print("\n样例（gold 行/列/值 → 模型用的行/列/值）：")
for name in ("case1", "case2"):
    print(" ", name)
    for e in examples[name]:
        print("   %-24s gold[%s|%s]=%s  used[%s|%s]=%s  %s" % e)
