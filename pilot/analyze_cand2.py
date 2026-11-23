# -*- coding: utf-8 -*-
"""
cand2（亚军候选）的配对确认。

三件事必须同时报，缺一个就会误读：
  1. EM 与配对 McNemar（同一批样本、同一次 prefill）
  2. 作用率：cand2 到底改变了多少样本的保留集（不触发时零结果无意义）
  3. 结构层：操作数覆盖与代价的变化

用法：python analyze_cand2.py ../results/cand2_confirm_rescored.json
"""
import json
import re
import sys
import numpy as np
from math import comb

from common import build_dataset
from anchor_auto import extract
from analyze_slice import nums

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")


def em_close(pred, gold):
    if pred is None:
        return False
    return any(abs(p - gold) <= max(0.01, 0.005 * abs(gold))
               for p in (pred, pred / 100.0, pred * 100.0))


def ci(v, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def mcnemar(a, b):
    x = int((a & ~b).sum())
    y = int((b & ~a).sum())
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/cand2_confirm_rescored.json"
    d = json.load(open(path, encoding="utf-8"))
    seq = {}
    for r in d["records"]:
        seq.setdefault(r["cond"], []).append(r)
    conds = list(seq)
    n = min(len(v) for v in seq.values())
    ok = {c: np.array([em_close(seq[c][i]["pred"], seq[c][i]["gold"]) for i in range(n)])
          for c in conds}

    print("== 1. EM（n=%d，同批样本）==" % n)
    for c in conds:
        lo, hi = ci(ok[c])
        print("  %-12s %.3f  95%%CI [%.3f, %.3f]" % (c, ok[c].mean(), lo, hi))

    if "auto_n@20" in ok and "auto_f@20" in ok:
        x, y, p = mcnemar(ok["auto_n@20"], ok["auto_f@20"])
        print("\n  配对：关 cand2 独对 %d | 开 cand2 独对 %d | McNemar p=%.4f" % (x, y, p))
        if "full" in ok:
            x2, y2, p2 = mcnemar(ok["full"], ok["auto_f@20"])
            print("  配对：全上下文独对 %d | cand2 独对 %d | p=%.4f" % (x2, y2, p2))

    # ---- 2/3. 结构层与作用率（与生成结果无关） ----
    ds = build_dataset("../data/dev.json")
    raw = {e.get("id"): e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
    changed = 0
    cov = {False: [], True: []}
    cnt = 0
    for smp in ds:
        if cnt >= n:
            break
        qa = (raw.get(smp["id"], {}) or {}).get("qa") or {}
        if not SINGLE.match((qa.get("program") or "").strip()):
            continue
        if not smp["group"].get("V"):
            continue
        a, _ = extract(smp["segs"], smp["question"], cand2=False)
        b, _ = extract(smp["segs"], smp["question"], cand2=True)
        if set(a) != set(b):
            changed += 1
        table = set()
        for s in smp["segs"]:
            if s["kind"] == "cell":
                table |= nums(s["text"])
        want = nums(qa.get("program") or "") & table
        for flag, idx in ((False, a), (True, b)):
            if want:
                kept = nums(" ".join(smp["segs"][i]["text"] for i in idx))
                cov[flag].append(len(kept & want) / len(want))
        cnt += 1

    print("\n== 2. 作用率 ==")
    print("  cand2 改变了保留集的样本：%d / %d (%.1f%%)" % (changed, cnt, 100.0 * changed / max(1, cnt)))
    print("\n== 3. 结构层（同批样本）==")
    for flag, name in ((False, "关 cand2"), (True, "开 cand2")):
        v = np.array(cov[flag])
        if len(v):
            print("  %-8s 全操作数 %.3f  平均覆盖 %.3f" % (
                name, (v >= 0.999).mean(), v.mean()))


if __name__ == "__main__":
    main()
