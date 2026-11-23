"""探针：最小可区分绑定能把「整对代价」降多少、装入率能抬多少。

不进 GPU：只做分词与代价计算，用来判断是否值得改 arm_c 并重跑。
"""
import json
import sys

from transformers import AutoTokenizer

from arm_a import tokenize_sample
from common import build_dataset

TOK = AutoTokenizer.from_pretrained("E:/kv_models/Qwen2.5-3B-Instruct")
FORCE = json.load(open("../results/r1_force.json", encoding="utf-8"))
KEYS = {(k["key"][0], round(float(k["key"][1]), 6))
        for k in json.load(open("../results/dev126_keys.json", encoding="utf-8"))}


def seg_tokens(ids, a, b):
    return ids[a:b]


def contains(seq, cand):
    n = len(cand)
    return any(seq[i:i + n] == cand for i in range(len(seq) - n + 1))


def _ok(cand, full, tok):
    txt = tok.decode(cand).strip()
    if len(txt) < 2 or not any(ch.isalnum() for ch in txt):
        return False
    # 不能从数字中间切开：若首字符是数字，前一个 token 不能以数字结尾
    if txt[0].isdigit():
        prev = tok.decode(full[:len(full) - len(cand)]).strip()
        if prev and prev[-1].isdigit():
            return False
        # 只是一个数字碎片（如 "7"）也不接受：原段里若有四位年份，要求至少保留四位
        digits = "".join(ch for ch in txt if ch.isdigit())
        if len(digits) < 4 and any(len(w) == 4 for w in
                                  "".join(tok.decode(full).replace(",", " ").split())):
            return False
    return True


def min_suffix(ids, others, tok, min_save=2):
    """最短可区分后缀：与同组其它段都不重复，且不切碎数字/符号。"""
    full = ids
    for m in range(1, len(full) + 1):
        cand = full[-m:]
        if not _ok(cand, full, tok):
            continue
        if all(not contains(o, cand) for o in others):
            if len(full) - m >= min_save:
                return m, tok.decode(cand).strip()
            return len(full), tok.decode(full).strip()
    return len(full), tok.decode(full).strip()


def main():
    ds = build_dataset("../data/dev.json")
    rows = []
    examples = []
    for d in ds:
        try:
            key = (d["id"], round(float(d["exe_ans"]), 6))
        except (TypeError, ValueError):
            continue
        if key not in KEYS:
            continue
        ids, sp = tokenize_sample(TOK, d["segs"])
        n_doc = len(ids)
        fp = FORCE.get("%s|%.6f" % key) or FORCE.get(d["id"])
        if not fp:
            continue
        want = [(int(r), int(c)) for r, c in fp]

        row_all = [ids[a:b] for s, (a, b) in zip(d["segs"], sp)
                   if s.get("kind") == "row_header" and a >= 0]
        col_all = [ids[a:b] for s, (a, b) in zip(d["segs"], sp)
                   if s.get("kind") in ("col_name", "col_header", "unit") and a >= 0]

        full = mini = 0
        for s, (a, b) in zip(d["segs"], sp):
            if a < 0 or b <= a:
                continue
            k = s.get("kind")
            if k == "cell" and (s.get("row"), s.get("col")) in want:
                full += b - a; mini += b - a
            elif k == "row_header" and any(s.get("row") == r for r, _ in want):
                full += b - a
                others = [o for o in row_all if o != ids[a:b]]
                m, txt = min_suffix(ids[a:b], others, TOK)
                mini += m
                if len(examples) < 6:
                    examples.append(("row", TOK.decode(ids[a:b]).strip(), txt))
            elif k in ("col_name", "col_header", "unit") and \
                    any(s.get("col") == c for _, c in want):
                full += b - a
                others = [o for o in col_all if o != ids[a:b]]
                m, txt = min_suffix(ids[a:b], others, TOK)
                mini += m
                if len(examples) < 6:
                    examples.append(("col", TOK.decode(ids[a:b]).strip(), txt))
        if any(r["id"] == d["id"] and r["n"] == n_doc for r in rows):
            continue
        rows.append(dict(id=d["id"], n=n_doc, full=full, mini=mini))

    def frac_fit(budget_pct, key):
        ok = sum(1 for r in rows if r[key] <= max(8, int(budget_pct / 100 * r["n"])))
        return ok / len(rows)

    cost_f = sorted(r["full"] for r in rows)
    cost_m = sorted(r["mini"] for r in rows)
    print("样本 %d 条" % len(rows))
    print("整对代价 中位：原 %d → 最小绑定 %d（降 %.0f%%）"
          % (cost_f[len(cost_f) // 2], cost_m[len(cost_m) // 2],
             100 * (1 - cost_m[len(cost_m) // 2] / cost_f[len(cost_f) // 2])))
    print("均值：%.1f → %.1f" % (sum(cost_f) / len(cost_f), sum(cost_m) / len(cost_m)))
    print("\n%-6s %-10s %-10s" % ("预算", "原装入率", "最小绑定居入率"))
    for b in (2, 3, 5, 10, 20):
        print("%-6s %.1f%%      %.1f%%" % ("%d%%" % b, 100 * frac_fit(b, "full"),
                                           100 * frac_fit(b, "mini")))
    print("\n缩写示例：")
    for k, orig, short in examples:
        print("  [%s] %-42s → %s" % (k, orig[:42], short))


if __name__ == "__main__":
    main()
