# -*- coding: utf-8 -*-
"""结构感知的成对打分器：在编码器之上显式建模「两格之间的关系」。

与 R1（只用 [CLS] 过 MLP）的区别：
  R1:      [CLS] ────────────────► MLP ► logit
  本文:    [CLS] , hA , hB , |hA-hB| , hA⊙hB , 结构标量 ──► MLP ► logit
其中 hA/hB 是两个格子各自在编码器输出上的池化向量（按 token 位置取掩码平均）。

动机：R1 的错误分析显示它常选「同一属性、不同实体」的两格（两个 low、两个
notional/contractamount、class b-1 与 b-4），而题目要的是不同属性。这个「互补性」
是两格之间的关系，用一个池化后的 [CLS] 去隐式学，样本效率很低；显式给出
差值/乘积项，等于把归纳偏置写进结构。

消融开关：
  --no_inter   去掉 |hA-hB| 与 hA⊙hB（只留 hA、hB）
  --no_struct  去掉 11 维结构标量
  --cls_only   退化为 R1（复现基线用）
"""
import argparse
import json
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, ".")
from r1_feat import pair_features, rule_and_ind, mcn  # noqa: E402
from ground import tokens  # noqa: E402

MAX_Q, MAX_C = 64, 32
NF = 11


class PairDS(Dataset):
    def __init__(self, rows, tok, train=False, max_pairs=16, seed=0):
        rng = random.Random(seed)
        self.tok = tok
        self.items = []
        for r in rows:
            ps = r["pairs"]
            if train and len(ps) > max_pairs:
                pos = [p for p in ps if p[2]]
                neg = [p for p in ps if not p[2]]
                ps = pos + rng.sample(neg, min(len(neg), max_pairs - len(pos)))
            if train and not any(p[2] for p in ps):
                continue
            gmax = max(c["gscore"] for c in r["cands"]) or 1.0
            qw = set(tokens(r["q"]))
            fs = [pair_features(r["cands"], i, j, r["q"], gmax, qw) for i, j, _ in ps]
            self.items.append((r, ps, fs))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        r, ps, fs = self.items[k]
        t = self.tok
        seqs, spans = [], []
        for i, j, _ in ps:
            qa = t(r["q"], add_special_tokens=False)["input_ids"][:MAX_Q]
            ca = t(r["cands"][i]["txt"], add_special_tokens=False)["input_ids"][:MAX_C]
            cb = t(r["cands"][j]["txt"], add_special_tokens=False)["input_ids"][:MAX_C]
            ids = [t.cls_token_id] + qa + [t.sep_token_id] + ca + [t.sep_token_id] + cb + [t.sep_token_id]
            a0 = 1 + len(qa) + 1
            a1 = a0 + len(ca)
            b0 = a1 + 1
            b1 = b0 + len(cb)
            seqs.append(ids)
            spans.append((a0, a1, b0, b1))
        return seqs, [p[2] for p in ps], fs, spans


def collate(batch, pad):
    B = len(batch)
    K = max(len(b[0]) for b in batch)
    L = max(len(s) for b in batch for s in b[0])
    ids = torch.full((B, K, L), pad, dtype=torch.long)
    am = torch.zeros((B, K, L), dtype=torch.long)
    pm = torch.zeros((B, K), dtype=torch.bool)
    y = torch.zeros((B, K))
    F = torch.zeros((B, K, NF))
    SP = torch.zeros((B, K, 4), dtype=torch.long)
    for i, (seqs, yy, fs, spans) in enumerate(batch):
        for j, s in enumerate(seqs):
            ids[i, j, :len(s)] = torch.tensor(s)
            am[i, j, :len(s)] = 1
            pm[i, j] = True
            y[i, j] = yy[j]
            F[i, j] = torch.tensor(fs[j])
            SP[i, j] = torch.tensor(spans[j])
    return ids, am, pm, y, F, SP


def masked_mean(h, am, a0, a1):
    """按 [a0, a1) 的 token 位置做掩码平均。"""
    B, K, L, H = h.shape
    idx = torch.arange(L, device=h.device).view(1, 1, L)
    m = ((idx >= a0.unsqueeze(-1)) & (idx < a1.unsqueeze(-1))).float()
    m = m.unsqueeze(-1)                      # B,K,L,1
    return (h * m).sum(2) / m.sum(2).clamp(min=1.0)


class StructCE(nn.Module):
    def __init__(self, base, use_inter=True, use_struct=True, cls_only=False, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.use_inter, self.use_struct, self.cls_only = use_inter, use_struct, cls_only
        if cls_only:
            nf = (NF if use_struct else 0)          # 退化为 R1：只有 [CLS] + 结构标量
        else:
            nf = 2 * h + (2 * h if use_inter else 0) + (NF if use_struct else 0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h + nf, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, ids, am, F, SP):
        B, K, L = ids.shape
        o = self.enc(input_ids=ids.view(B * K, L),
                     attention_mask=am.view(B * K, L)).last_hidden_state
        o = o.view(B, K, L, -1)
        cls = o[:, :, 0]
        parts = [cls]
        if not self.cls_only:
            a0, a1, b0, b1 = (SP[..., t] for t in range(4))
            hA = masked_mean(o, am, a0, a1)
            hB = masked_mean(o, am, b0, b1)
            parts += [hA, hB]
            if self.use_inter:
                parts += [(hA - hB).abs(), hA * hB]
        if self.use_struct:
            parts.append(F)
        x = torch.cat(parts, dim=-1)
        return self.head(self.drop(x)).squeeze(-1)


def evaluate(model, dl):
    model.eval()
    hits, n = 0, 0
    rec = []
    with torch.no_grad():
        for ids, am, pm, y, F, SP in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda(), SP.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            pick = s.argmax(1)
            for i in range(s.shape[0]):
                v = float(y[i, pick[i]])
                hits += v
                rec.append(v)
                n += 1
    return hits / max(1, n), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_inter", action="store_true")
    ap.add_argument("--no_struct", action="store_true")
    ap.add_argument("--cls_only", action="store_true")
    ap.add_argument("--tag", default="r1s")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = json.load(open("../results/r1_pairs_train_full.json", encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev_full.json", encoding="utf-8"))
    te = json.load(open("../results/xds_pairs_test.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id or 0

    docs = sorted({r["doc"] for r in tr})
    rng = random.Random(0)
    rng.shuffle(docs)
    vd = set(docs[:len(docs) // 10])
    tr2 = [r for r in tr if r["doc"] not in vd]
    va = [r for r in tr if r["doc"] in vd]
    print("训练 %d / 验证 %d / dev %d / test %d" % (len(tr2), len(va), len(dv), len(te)), flush=True)

    mk = lambda rows, train=False: PairDS(rows, tok, train=train, seed=args.seed)
    ds_tr = mk(tr2, True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       collate_fn=lambda b: collate(b, pad))
    dl_va = DataLoader(mk(va), batch_size=8, shuffle=False, collate_fn=lambda b: collate(b, pad))
    dl_dv = DataLoader(mk(dv), batch_size=8, shuffle=False, collate_fn=lambda b: collate(b, pad))
    dl_te = DataLoader(mk(te), batch_size=8, shuffle=False, collate_fn=lambda b: collate(b, pad))

    model = StructCE(args.base, use_inter=not args.no_inter,
                     use_struct=not args.no_struct, cls_only=args.cls_only).cuda()
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.15)

    npos = sum(p[2] for it in ds_tr.items for p in it[1])
    ntot = sum(len(it[1]) for it in ds_tr.items)
    pw = torch.tensor([(ntot - npos) / max(1, npos)], device="cuda")
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    print("训练对 %d，正例 %d（%.1f%%），pos_weight=%.1f"
          % (ntot, npos, 100 * npos / max(1, ntot), pw.item()), flush=True)

    best = (-1, None, -1)
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for bi, (ids, am, pm, y, F, SP) in enumerate(dl_tr):
            s = model(ids.cuda(), am.cuda(), F.cuda(), SP.cuda())
            loss = lossf(s[pm.cuda()], y.cuda()[pm.cuda()])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            if bi % 50 == 0:
                print("  ep%d %d/%d loss %.4f" % (ep + 1, bi, len(dl_tr), tot / max(1, n)), flush=True)
        v, _ = evaluate(model, dl_va)
        print("epoch %d：验证 %.4f" % (ep + 1, v), flush=True)
        if v > best[0]:
            best = (v, {k: t.cpu().clone() for k, t in model.state_dict().items()}, ep + 1)

    model.load_state_dict(best[1])
    print("取 epoch %d（验证 %.4f）" % (best[2], best[0]), flush=True)
    hd, rd = evaluate(model, dl_dv)
    ht, rt = evaluate(model, dl_te)
    IND, JR = rule_and_ind(dv)
    INDT, JRT = rule_and_ind(te)
    print("\n=== %s ===" % args.tag)
    print("dev  114 题：%.3f（R1 0.570 / 规则 0.447）" % hd)
    print("test 182 题：%.3f（R1 0.511 / 规则 0.363）" % ht)
    x, y, p = mcn([bool(v) for v in rd], JR)
    print("  dev  vs 规则：+%d/-%d p=%.4g" % (x, y, p))
    torch.save({"base": args.base, "state": model.state_dict(),
                "args": vars(args)}, "../results/%s.pt" % args.tag)


if __name__ == "__main__":
    main()
