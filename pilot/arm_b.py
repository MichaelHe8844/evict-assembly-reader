# -*- coding: utf-8 -*-
"""
Arm B：注意力驱动的驱逐，对比「数值留存」与「锚点组完整留存」。

策略
----
snapkv    : 用最后 W 个 query 位置的注意力作为重要性（SnapKV 的 observation window）
h2o       : 用全部 query 位置的累计注意力作为重要性（H2O）
oracle    : 先无条件保留整个锚点组，剩余预算按 snapkv 分数填 —— 这是「数值锚点」
            方法的上界，用 gold 锚点构造，只能作为 oracle 参照，不能当结果。

额外输出「注意力富集比」：
    enrich(c) = 成分 c 的注意力占比 / 成分 c 的 token 占比
若单位(U)的富集比显著 < 1，说明模型本身就不怎么关注量纲 ——
这正是注意力类压缩必然丢掉单位的原因，也是「数值锚点」需要外部结构信号的理由。
"""
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import build_dataset
from arm_a import tokenize_sample, comp_status
from model_utils import load_model


@torch.no_grad()
def importance(model, ids, window=64, n_ctx=None):
    """返回 (imp_snap, imp_h2o)，仅保留前 n_ctx 个(上下文)位置。"""
    x = torch.tensor([ids], device=model.device)
    out = model(input_ids=x, output_attentions=True, use_cache=False)
    atts = out.attentions                      # tuple(L_layer) of (1,H,Q,K)
    L = len(ids)
    dev = atts[0].device
    snap = torch.zeros(L, dtype=torch.float32, device=dev)
    h2o = torch.zeros(L, dtype=torch.float32, device=dev)
    for a in atts:
        a = a[0].float()                       # (H,Q,K)
        h2o += a.sum(dim=(0, 1))               # 所有 query 累计
        snap += a[:, -window:, :].sum(dim=(0, 1))
    del atts, out
    torch.cuda.empty_cache()
    k = n_ctx if n_ctx else L
    return snap[:k].cpu().numpy(), h2o[:k].cpu().numpy()


def mask_topk(score, L, B, force=None):
    m = np.zeros(L, dtype=bool)
    sink = min(4, L)
    m[:sink] = True
    if force:
        for i in force:
            m[i] = True
    need = max(0, min(L, int(B)) - int(m.sum()))
    if need > 0:
        cand = np.where(~m)[0]
        if len(cand) > 0:
            top = cand[np.argsort(-score[cand])[:need]]
            m[top] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--maxlen", type=int, default=1600)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--out", default="../results/arm_b.json")
    ap.add_argument("--query_aware", action="store_true",
                    help="把问题接在上下文后一起编码（否则为 query-agnostic 压缩）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.load_4bit).eval()
    print("model loaded:", model.config.model_type,
          "layers", model.config.num_hidden_layers,
          "heads", model.config.num_attention_heads)

    ds = build_dataset(args.data)
    print(f"候选 {len(ds)}")

    budgets = [0.10, 0.20, 0.30, 0.50]
    policies = ["snapkv", "h2o", "oracle"]

    cnt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    enrich = defaultdict(list)
    used, lens = 0, []

    for n, d in enumerate(ds):
        if used >= args.limit:
            break
        ctx, spans = tokenize_sample(tok, d["segs"])
        L = len(ctx)
        if L < 64 or L > args.maxlen:
            continue

        if args.query_aware:
            q = "\n\nQuestion: %s\nAnswer:" % d["question"]
            ids = ctx + tok(q, add_special_tokens=False)["input_ids"]
        else:
            ids = ctx

        snap, h2o = importance(model, ids, args.window, n_ctx=L)
        used += 1
        lens.append(L)

        # 注意力富集比：= 注意力占比 / token 占比；背景为所有非锚点 token 的中位数
        tot = snap.sum() + 1e-9
        enr_tok = snap * L / tot
        anchor_toks = set()
        for c in ("V", "R", "C", "U"):
            for i in d["group"].get(c, []):
                a, b = spans[i]
                if a >= 0:
                    anchor_toks.update(range(a, b))
        bg = [enr_tok[t] for t in range(L) if t not in anchor_toks]
        if bg:
            enrich["BG"].append(float(np.median(bg)))

        for c in ("V", "R", "C", "U"):
            idxs = d["group"].get(c, [])
            if not idxs:
                continue
            tk = sum(spans[i][1] - spans[i][0] for i in idxs if spans[i][0] >= 0)
            if tk <= 0:
                continue
            s = 0.0
            for i in idxs:
                a, b = spans[i]
                if a >= 0:
                    s += snap[a:b].sum()
            enrich[c].append((s / tot) / (tk / L))

        force = [t for i in d["group"]["V"] + d["group"]["R"] + d["group"]["C"] + d["group"]["U"]
                 for t in range(*spans[i]) if spans[i][0] >= 0]

        for frac in budgets:
            B = int(frac * L)
            for pol in policies:
                score = snap if pol != "h2o" else h2o
                fz = force if pol == "oracle" else None
                m = mask_topk(score, L, B, force=fz)
                st = {c: comp_status(m, spans, d["group"].get(c, []))
                      for c in ("V", "R", "C", "U")}
                v_all, _ = st["V"]
                anr = all(v[0] is None or v[0] for v in st.values())
                k = (pol, frac)
                cnt[k]["V"][0] += int(v_all); cnt[k]["V"][1] += 1
                cnt[k]["ANR"][0] += int(anr); cnt[k]["ANR"][1] += 1
                if v_all:
                    cnt[k]["frag|V"][0] += int(not anr); cnt[k]["frag|V"][1] += 1
                for c in ("R", "C", "U"):
                    if st[c][0] is not None:
                        cnt[k][c][0] += int(st[c][0]); cnt[k][c][1] += 1

        if used % 20 == 0:
            print(f"  {used} samples, L={L}, vram {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def rate(k, m):
        h, t = cnt[k][m]
        return h / t if t else float("nan")

    print(f"\n样本 {used}  长度 median {np.median(lens):.0f} max {max(lens)}")
    print("\n%-8s %6s %6s %6s %6s %6s %6s %10s" % (
        "policy", "budget", "NumR", "ANR", "R", "C", "U", "P(frag|V)"))
    rows = []
    for pol in policies:
        for frac in budgets:
            r = dict(policy=pol, budget=frac, NumR=rate((pol, frac), "V"),
                     ANR=rate((pol, frac), "ANR"),
                     R=rate((pol, frac), "R"), C=rate((pol, frac), "C"),
                     U=rate((pol, frac), "U"),
                     frag_cond=rate((pol, frac), "frag|V"))
            rows.append(r)
            print("%-8s %5.0f%% %6.3f %6.3f %6.3f %6.3f %6.3f %9.3f" % (
                pol, frac * 100, r["NumR"], r["ANR"], r["R"], r["C"], r["U"], r["frag_cond"]))

    print("\n注意力富集比 (1.0=与 token 占比相当):")
    for c in ("BG", "V", "R", "C", "U"):
        if enrich[c]:
            print("  %s: median %.2f  mean %.2f  n=%d" % (
                c, np.median(enrich[c]), np.mean(enrich[c]), len(enrich[c])))

    json.dump({"rows": rows, "n": used,
               "enrich": {c: float(np.median(v)) for c, v in enrich.items() if v}},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n写入", args.out)


if __name__ == "__main__":
    main()
