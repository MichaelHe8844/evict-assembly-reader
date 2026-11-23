# -*- coding: utf-8 -*-
"""
Arm A：模型无关的「锚点组碎裂」分析。

关键指标定义
------------
一个数值单元格是原子单位：'11503' 会被切成 2-3 个 token，任一 token 丢失
数值即损坏。因此「数值留存 NumR」要求 V 的全部 token 都在。

锚点组 ANR 有两种判据：
  strict : 每个成分的全部 token 都在（单位被截断成 'bill' 也算损坏）
  lenient: 每个成分至少有 1 个 token 在（部分保留仍可恢复语义）

主报 strict，lenient 作稳健性检验。

核心指标是条件碎裂率：
  P(frag | NumR) = P(¬ANR ∧ NumR) / P(NumR)
即「数值活下来了，但它和主体/列名/单位的绑定断了」的比例。
"""
import json
import argparse
import numpy as np
from collections import defaultdict

from common import build_dataset


def tokenize_sample(tok, segs):
    ids, spans = [], []
    for s in segs:
        t = tok(s["text"], add_special_tokens=False)["input_ids"]
        if not t:
            spans.append((-1, -1))
            continue
        spans.append((len(ids), len(ids) + len(t)))
        ids.extend(t)
    return ids, spans


def retained_mask(policy, L, B, rng):
    m = np.zeros(L, dtype=bool)
    B = max(8, min(L, int(B)))
    if policy == "streaming":
        k = min(4, L)
        m[:k] = True
        m[max(k, L - (B - k)):] = True
    elif policy == "head_tail":
        h = B // 2
        m[:h] = True
        m[L - (B - h):] = True
    elif policy == "random":
        m[rng.choice(L, size=B, replace=False)] = True
    elif policy == "full":
        m[:] = True
    else:
        raise ValueError(policy)
    return m


def comp_status(m, spans, idxs):
    """返回 (all_tokens_kept, any_token_kept)；成分不存在或分词为空时返回 (None, None)。"""
    if not idxs:
        return None, None
    all_ok, any_ok, valid = True, False, 0
    for i in idxs:
        a, b = spans[i]
        if a < 0 or b <= a:          # 空片段不携带信息，不应判为损坏
            continue
        valid += 1
        seg = m[a:b]
        if seg.all():
            any_ok = True
        else:
            all_ok = False
            if seg.any():
                any_ok = True
    if valid == 0:
        return None, None
    return all_ok, any_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="../results/arm_a.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    ds = build_dataset(args.data)[: args.limit]
    print(f"样本 {len(ds)}")

    budgets = [0.05, 0.10, 0.20, 0.30, 0.50]
    policies = ["streaming", "head_tail", "random"]

    # counters[(policy,budget)][metric] = [hit, total]
    cnt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    lens, dists, vlen = [], [], []

    for n, d in enumerate(ds):
        ids, spans = tokenize_sample(tok, d["segs"])
        L = len(ids)
        if L < 64:
            continue
        lens.append(L)
        vlen.append(sum(spans[i][1] - spans[i][0] for i in d["group"]["V"] if spans[i][0] >= 0))

        pos = [spans[i][0] for c in ("V", "R", "C", "U")
               for i in d["group"].get(c, []) if spans[i][0] >= 0]
        if len(pos) >= 2:
            dists.append(max(pos) - min(pos))

        for frac in budgets:
            B = int(frac * L)
            for pol in policies:
                reps = args.reps if pol == "random" else 1
                rng = np.random.default_rng(1234 + n)
                for _ in range(reps):
                    m = retained_mask(pol, L, B, rng)
                    st = {c: comp_status(m, spans, d["group"].get(c, []))
                          for c in ("V", "R", "C", "U")}

                    v_all, _ = st["V"]
                    anr_s = all(v[0] is None or v[0] for v in st.values())
                    anr_l = all(v[1] is None or v[1] for v in st.values())

                    def bump(metric, cond):
                        cnt[(pol, frac)][metric][0] += int(cond)
                        cnt[(pol, frac)][metric][1] += 1

                    bump("V", v_all)
                    bump("ANRs", anr_s)
                    bump("ANRl", anr_l)
                    bump("frag", v_all and not anr_s)
                    for c in ("R", "C", "U"):
                        if st[c][0] is not None:      # 仅在成分存在时计入分母
                            bump(c, st[c][0])
                    # 分母受限：仅在 V 存活的样本里统计碎裂
                    if v_all:
                        bump("frag|V", not anr_s)

        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(ds)}")

    def rate(pol, frac, metric):
        h, t = cnt[(pol, frac)][metric]
        return h / t if t else float("nan")

    print("\n上下文长度 token: median %.0f  p90 %.0f  max %d" % (
        np.median(lens), np.percentile(lens, 90), max(lens)))
    print("绑定距离 token  : median %.0f  p90 %.0f  max %d" % (
        np.median(dists), np.percentile(dists, 90), max(dists)))
    print("数值单元格 token 数: mean %.2f  median %.0f" % (np.mean(vlen), np.median(vlen)))
    print("  参考: 若均匀随机保 50%%，一个 %0.1f-token 数字的存活率约 %.3f"
          % (np.mean(vlen), 0.5 ** np.mean(vlen)))

    hdr = ("policy", "budget", "NumR", "ANR", "R", "C", "U", "P(frag|V)")
    print("\n%-9s %6s %6s %6s %6s %6s %6s %10s" % hdr)
    rows = []
    for pol in policies:
        for frac in budgets:
            r = dict(policy=pol, budget=frac,
                     NumR=rate(pol, frac, "V"),
                     ANR=rate(pol, frac, "ANRs"),
                     R=rate(pol, frac, "R"),
                     C=rate(pol, frac, "C"),
                     U=rate(pol, frac, "U"),
                     frag_cond=rate(pol, frac, "frag|V"),
                     ANR_lenient=rate(pol, frac, "ANRl"))
            rows.append(r)
            print("%-9s %5.0f%% %6.3f %6.3f %6.3f %6.3f %6.3f %9.3f" % (
                pol, frac * 100, r["NumR"], r["ANR"], r["R"], r["C"], r["U"], r["frag_cond"]))

    json.dump({"rows": rows, "n": len(lens),
               "len_median": float(np.median(lens)),
               "dist_median": float(np.median(dists)),
               "dist_p90": float(np.percentile(dists, 90)),
               "value_tokens_mean": float(np.mean(vlen))},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n写入", args.out)


if __name__ == "__main__":
    main()
