# -*- coding: utf-8 -*-
"""按"操作数是否被定位到"分解算术错误。

与 5.12 的差别：那一节按"模型用了一个表里别的格"分类，会把
「用了 gold + 第二个操作数错」算成一次选错；这里改成**操作数级**：
每个 gold 实参（表格里的）是否被模型表达式用到。

于是每一道题在每个条件下有三种状态：
  full  → 全部命中 / 部分命中 / 全未命中     （定位能力，与压缩无关）
  压缩  → 同上                              （定位在压缩下是否稳定）

用户要的两类：
  case 1：两个条件都错，且压缩后仍"全未命中" → 定位能力不足
  case 2：full 对、压缩错、gold 值仍保留、压缩后"全未命中/部分命中" → 压缩诱导的定位漂移
"""
import json
import re
from collections import Counter

NUM = re.compile(r"-?\d[\d,]*\.?\d*%?")
SINGLE = re.compile(r"^([a-z]+)\(([^()]*)\)$")
CONST = {1.0, 100.0, 1000.0, 0.01, 0.1, 0.001}


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
    return [v for v in (to_float(m) for m in NUM.findall(s or "")) if v is not None]


def close(a, b, rel=0.005):
    return a is not None and b is not None and abs(a - b) <= rel * max(1e-9, abs(b))


def grid(entry):
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
                            col=cols[c] if c < len(cols) else "", raw=str(row[c])))
    return out


def gold_args(entry):
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
    return (m.group(1), args) if args else None


def gold_cells(entry):
    """每个实参对应的单元格；实参不在表里（来自正文）则记 None。"""
    g = gold_args(entry)
    if not g:
        return None
    gs = grid(entry)
    rows = set()
    for k in (entry["qa"].get("gold_inds") or {}):
        if k.startswith("table_"):
            try:
                rows.add(int(k.split("_")[1]))
            except ValueError:
                pass
    out = []
    for a in g[1]:
        hit = [c for c in gs if (not rows or c["r"] in rows) and close(c["v"], a, 1e-3)]
        if not hit:
            hit = [c for c in gs if close(c["v"], a, 1e-3)]
        out.append(hit[0] if hit else None)
    return out


def used_numbers(rec):
    return [v for v in numbers(rec.get("text", "")) if v not in CONST]


def hit_state(rec, cells):
    """返回 ('all'|'part'|'none', 命中数, 总数)。只看能定位到表格的实参。"""
    cs = [c for c in cells if c is not None]
    if not cs:
        return None
    us = used_numbers(rec)
    k = sum(1 for c in cs if any(close(u, c["v"], 1e-3) for u in us))
    return ("all" if k == len(cs) else ("none" if k == 0 else "part")), k, len(cs)


def load_pairs():
    seen, pairs = set(), []
    for f in ("../results/cand2_confirm_rescored.json",
              "../results/arm_c_real_rescored.json"):
        d = json.load(open(f, encoding="utf-8"))
        by = {}
        for r in d["records"]:
            by.setdefault((r["id"], round(float(r["gold"]), 6)), {})[r["cond"]] = r
        for k, c in by.items():
            if k in seen or "full" not in c or "auto_f@20" not in c:
                continue
            seen.add(k)
            pairs.append((k, c["full"], c["auto_f@20"]))
    return pairs


if __name__ == "__main__":
    raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
    pairs = load_pairs()

    stat = Counter()
    rows = []
    for k, f, c in pairs:
        entry = raw.get(k[0], {})
        cells = gold_cells(entry)
        if cells is None:
            stat["程序非单步"] += 1
            continue
        hs_f = hit_state(f, cells)
        hs_c = hit_state(c, cells)
        if hs_f is None:
            stat["实参不在表内"] += 1
            continue
        st_f, st_c = hs_f[0], hs_c[0]
        key = "full=%s|comp=%s" % (st_f, st_c)
        stat[key] += 1
        rows.append(dict(sk="full=%s|comp=%s" % (st_f, st_c), key=k, id=k[0], gold=k[1], f=f, c=c, cells=cells,
                         st_f=st_f, st_c=st_c,
                         ret=bool(c.get("num")), air=c.get("air")))

    n = len(rows)
    print("可用样本 %d（程序非单步 %d、实参不在表内 %d）" % (
        n, stat["程序非单步"], stat["实参不在表内"]))
    print("\n【定位状态 × 配对】")
    hdr = ["full=all|comp=all", "full=all|comp=part", "full=all|comp=none",
           "full=part|comp=all", "full=part|comp=part", "full=part|comp=none",
           "full=none|comp=all", "full=none|comp=part", "full=none|comp=none"]
    for h in hdr:
        if stat[h]:
            s = [r for r in rows if r["sk"] == h]
            em_c = sum(r["c"]["correct"] for r in s) / len(s)
            ret = sum(r["ret"] for r in s) / len(s)
            print("  %-26s n=%-4d 压缩后EM=%.3f  gold值保留=%.3f" % (h, len(s), em_c, ret))

    # 用户要的两类
    case1 = [r for r in rows if r["st_f"] == "none" and r["st_c"] == "none"]
    case2 = [r for r in rows if r["f"]["correct"] and not r["c"]["correct"]
             and r["ret"] and r["st_c"] != "all"]
    print("\n【案例 1】full 与压缩后都「全未命中」：%d 题（占可用 %d 的 %.1f%%）" % (
        len(case1), n, 100 * len(case1) / n))
    print("          其中压缩后仍然错误的 %d 题，说明定位能力本身是瓶颈" %
          sum(1 for r in case1 if not r["c"]["correct"]))
    print("【案例 2】full 对、压缩错、gold 值仍在、压缩后定位退化：%d 题" % len(case2))
    for r in case2[:10]:
        print("          %-24s gold=%s  压缩后用到=%s" % (
            r["id"][:24], [round(c["v"], 4) if c else None for c in r["cells"]],
            [round(v, 4) for v in used_numbers(r["c"])[:3]]))

    # 定位漂移率：在 gold 值保留的样本里，full 命中全部而 compressed 未命中全部
    keep = [r for r in rows if r["ret"]]
    print("\n【定位漂移】gold 值保留的 %d 题中：full 全命中 %d（%.1f%%），"
          "压缩后全命中 %d（%.1f%%），下降 %.1f 个百分点" % (
              len(keep),
              sum(1 for r in keep if r["st_f"] == "all"), 100 * sum(1 for r in keep if r["st_f"] == "all") / len(keep),
              sum(1 for r in keep if r["st_c"] == "all"), 100 * sum(1 for r in keep if r["st_c"] == "all") / len(keep),
              100 * (sum(1 for r in keep if r["st_f"] == "all") - sum(1 for r in keep if r["st_c"] == "all")) / len(keep)))
