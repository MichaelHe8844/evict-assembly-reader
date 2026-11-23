# -*- coding: utf-8 -*-
"""多步题的缓存指标探针：只测 ANR / 可定位召回 / 数值召回，不生成、不训练。

目的：验证"以四元组为保留单元"的装配在多步题（k>=2 个操作数）上仍然成立，
即随预算收紧，它保住的不是"某个数"，而是"每一个操作数连同它的行头、列名、单位"。

指标（全部是缓存侧判定，与读者无关）
  NumR : 所有操作数数值都出现在保留文本里
  R_B  : 每个操作数都能找到一个同值单元格，其数值格 + 行头 + 列名都在保留集内
  ANR  : R_B 且（该记录有单位时）单位片段也在保留集内

条件
  ours     本文装配（先数值后绑定 + 整段装入，候选来自 4.5 的抽取器）
  ours_old 同一装配但用旧顺序（表头优先）
  numonly  只保数值格
  row      只保数值格 + 行头
  col      只保数值格 + 列名/列头
  prefix   截断式基线（前 B 个 token）

用法
  python probe_multistep_cache.py --data ../data/dev.json --budgets 3,5,10,20 \
      --out ../results/multistep_cache_dev.json
"""
import argparse
import collections
import json
import re
import sys
import time

sys.path.insert(0, ".")
import arm_c                                        # noqa: E402
from arm_a import tokenize_sample                   # noqa: E402
from anchor_auto import extract                     # noqa: E402
from common import build_dataset                    # noqa: E402

NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def parse_num(t):
    m = NUM.search(str(t).replace("$", " ").replace("%", " "))
    if not m:
        return None
    try:
        return round(float(m.group(0).replace(",", "")), 4)
    except ValueError:
        return None


def gold_literals(prog):
    p = re.sub(r"const_\d+|#\d+", " ", prog or "")
    out = []
    for m in NUM.finditer(p):
        try:
            out.append(round(float(m.group(0).replace(",", "")), 4))
        except ValueError:
            pass
    return out


def n_ops(prog):
    p = re.sub(r"const_\d+|#\d+", " ", prog or "")
    return len(re.findall(r"[A-Za-z_]+\s*\(", p))


def keep_tokens(cond, tok, rec, frac):
    segs = rec["segs"]
    doc, spans = tokenize_sample(tok, segs)
    n = len(doc)
    if frac is None:
        return set(range(n))
    B = max(8, int(frac * n))
    if cond == "prefix":
        return set(range(min(B, n)))
    if cond == "oracle":
        cells = set()
        for g in rec["_golds"]:
            for i, sg in enumerate(segs):
                if sg["kind"] == "cell" and parse_num(sg["text"]) is not None \
                        and abs(parse_num(sg["text"]) - g) <= 1e-3 * max(1, abs(g)):
                    cells.add(i)
        want = set(cells)
        rows = {segs[i]["row"] for i in cells}
        cols = {segs[i]["col"] for i in cells}
        for i, sg in enumerate(segs):
            if sg["kind"] == "row_header" and sg["row"] in rows:
                want.add(i)
            if sg["kind"] in ("col_name", "col_header") and (sg["col"] in cols or sg["row"] == 0):
                want.add(i)
            if sg["kind"] == "unit":
                want.add(i)
        out = set()
        for i in sorted(want, key=lambda k: (0 if segs[k]["kind"] == "cell" else 1, k)):
            a, b = spans[i]
            if a >= 0 and b > a and len(out) + (b - a) <= B:
                out |= set(range(a, b))
        return out
    if cond in ("ours", "ours_old"):
        mc = max(1, int((0.85 * B) / 6))
        idx, _ = extract(segs, rec["question"], max_cells=mc)
        idx = [i for i in idx if spans[i][0] >= 0]
        arm_c.VALUE_FIRST = (cond == "ours")
        return set(arm_c.fit_budget(idx, segs, spans, B))
    kinds = {"numonly": ("cell",),
             "row": ("cell", "row_header"),
             "col": ("cell", "col_name", "col_header")}[cond]
    out = set()
    for i, s in enumerate(segs):
        if s["kind"] not in kinds:
            continue
        a, b = spans[i]
        if a < 0 or b <= a:
            continue
        if len(out) + (b - a) <= B:
            out |= set(range(a, b))
    return out


def segs_kept(segs, spans, kept):
    return {i for i, (a, b) in enumerate(spans)
            if a >= 0 and b > a and all(t in kept for t in range(a, b))}


def score(rec, tok, cond, frac):
    segs = rec["segs"]
    doc, spans = tokenize_sample(tok, segs)
    kept = keep_tokens(cond, tok, rec, frac)
    ks = segs_kept(segs, spans, kept)
    text = tok.decode([doc[i] for i in sorted(kept)]) if kept else ""

    cells = [(i, s["row"], s["col"], parse_num(s["text"]))
             for i, s in enumerate(segs) if s["kind"] == "cell"]
    rowh = {s["row"]: i for i, s in enumerate(segs) if s["kind"] == "row_header"}
    coln = {}
    for i, s in enumerate(segs):
        if s["kind"] in ("col_name", "col_header") and s["col"] not in coln:
            coln[s["col"]] = i
    units = [i for i, s in enumerate(segs) if s["kind"] == "unit"]

    nums_in_text = {parse_num(w) for w in NUM.findall(text.replace("$", " ").replace("%", " "))}
    nums_in_text.discard(None)

    golds = list(dict.fromkeys(rec["_golds"]))
    if not golds:
        return None
    numr = all(any(abs(g - v) <= 1e-3 * max(1, abs(g)) for v in nums_in_text) for g in golds)
    loc = True
    anchored = 0
    for g in golds:
        ok = False
        for i, r, c, v in cells:
            if v is None or abs(v - g) > 1e-3 * max(1, abs(g)):
                continue
            if i not in ks:
                continue
            if r in rowh and rowh[r] not in ks:
                continue
            if c in coln and coln[c] not in ks:
                continue
            ok = True
            break
        anchored += ok
        if not ok:
            loc = False
    unit_ok = True
    if rec.get("has_unit") and units:
        unit_ok = all(u in ks for u in units)
    return dict(numr=numr, loc=loc, anr=(loc and unit_ok),
                frac=anchored / len(golds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--budgets", default="3,5,10,20")
    ap.add_argument("--conds", default="ours,ours_old,oracle,prefix")
    ap.add_argument("--min_ops", type=int, default=2)
    ap.add_argument("--out", default="../results/multistep_cache_dev.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import warnings
    warnings.filterwarnings("ignore")
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")

    raw = {r["id"]: r for r in json.load(open(args.data, encoding="utf-8"))}
    rows, seen = [], set()
    for d in build_dataset(args.data):
        if d["id"] in seen:
            continue
        prog = ((raw.get(d["id"], {}).get("qa") or {}).get("program") or "")
        if n_ops(prog) < args.min_ops:
            continue
        g = gold_literals(prog)
        if len(g) < 2:
            continue
        seen.add(d["id"])
        d["_golds"] = g
        d["_ops"] = n_ops(prog)
        rows.append(d)
    print("多步题 %d 道（>=%d 步）" % (len(rows), args.min_ops), flush=True)

    out = {"n": len(rows), "conds": {}, "by_ops": {}}
    conds = args.conds.split(",")
    for b in args.budgets.split(","):
        frac = float(b) / 100.0
        t0 = time.time()
        for cond in conds:
            agg = collections.Counter()
            cut = collections.defaultdict(lambda: collections.Counter())
            for d in rows:
                r = score(d, tok, cond, frac)
                if r is None:
                    continue
                agg["n"] += 1
                for k in ("numr", "loc", "anr"):
                    agg[k] += r[k]
                agg["frac"] += r["frac"]
                key = "2步" if d["_ops"] == 2 else ">=3步"
                cut[key]["n"] += 1
                for k in ("numr", "loc", "anr"):
                    cut[key][k] += r[k]
                cut[key]["frac"] += r["frac"]
            n = max(1, agg["n"])
            out["conds"].setdefault(cond, {})[b] = {
                k: round(agg[k] / n, 3) for k in ("numr", "loc", "anr", "frac")}
            out["by_ops"].setdefault(cond, {})[b] = {
                key: {k: round(v[k] / max(1, v["n"]), 3) for k in ("numr", "loc", "anr", "frac")}
                for key, v in cut.items()}
        line = "  ".join("%s %.3f" % (c, out["conds"][c][b]["anr"]) for c in conds)
        print("预算 %-3s ANR: %s  (%.0fs)" % (b, line, time.time() - t0), flush=True)

    out["budgets"] = args.budgets
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n%-10s %6s %6s %6s %6s" % ("cond", "NumR", "定位", "ANR", "均值"))
    for cond in conds:
        for b in args.budgets.split(","):
            v = out["conds"][cond][b]
            print("%-10s@%-3s %6.3f %6.3f %6.3f %6.3f"
                  % (cond, b, v["numr"], v["loc"], v["anr"], v["frac"]))
    print("\n写入", args.out)


if __name__ == "__main__":
    main()
