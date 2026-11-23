# -*- coding: utf-8 -*-
"""R1c：结构感知的成对打分器。

= 语义编码器（问题 × 两格文本） + 结构特征向量 → MLP → logit

为什么要把两者拼起来：错误分析显示，纯语义编码器最大的失误是
「选同一属性、不同实体的两格」（两个 low、两个 notional、class b-1/b-4），
而题目要的是不同属性（high 对 low、分子对总额）。这个「互补性」是
结构关系（同行/同列、表头重叠、取值比），不是文本里读得出来的；
反过来，结构关系也判断不了「class a 才对应 class a common stock」，
那需要语义。所以两者必须各管一半。
"""
import argparse
import json
import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MAX_Q = 64
MAX_C = 32
NF = 11


def pair_features(cands, i, j, q, gmax, qw):
    a, b = cands[i], cands[j]
    ra, rb = set(a.get("row_t") or []), set(b.get("row_t") or [])
    ca, cb = set(a.get("col_t") or []), set(b.get("col_t") or [])
    inter = len(ra & rb) + len(ca & cb)
    uni = len(ra | rb) + len(ca | cb)
    ratio = abs(math.log((abs(a["v"]) + 1e-9) / (abs(b["v"]) + 1e-9)))
    return [
        a["gscore"] / gmax, b["gscore"] / gmax,
        1.0 if a["r"] == b["r"] else 0.0,
        1.0 if a["c"] == b["c"] else 0.0,
        min(ratio, 5.0) / 5.0,
        inter / max(1, uni),
        len(qw & ra) / max(1, len(ra)),
        len(qw & rb) / max(1, len(rb)),
        len(qw & ca) / max(1, len(ca)),
        len(qw & cb) / max(1, len(cb)),
        1.0 if (i == 0 or j == 0) else 0.0,
    ]


class QDS(Dataset):
    def __init__(self, rows, tok, train=False, max_pairs=16, seed=0):
        rng = random.Random(seed)
        self.tok = tok
        self.items = []
        from ground import tokens
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
        seqs = []
        for i, j, _ in ps:
            qa = t(r["q"], add_special_tokens=False)["input_ids"][:MAX_Q]
            ca = t(r["cands"][i]["txt"], add_special_tokens=False)["input_ids"][:MAX_C]
            cb = t(r["cands"][j]["txt"], add_special_tokens=False)["input_ids"][:MAX_C]
            seqs.append([t.cls_token_id] + qa + [t.sep_token_id] + ca +
                        [t.sep_token_id] + cb + [t.sep_token_id])
        return seqs, [p[2] for p in ps], fs, r["key"]


def collate(batch, pad):
    B = len(batch)
    K = max(len(b[0]) for b in batch)
    L = max(len(s) for b in batch for s in b[0])
    ids = torch.full((B, K, L), pad, dtype=torch.long)
    am = torch.zeros((B, K, L), dtype=torch.long)
    pm = torch.zeros((B, K), dtype=torch.bool)
    y = torch.zeros((B, K))
    F = torch.zeros((B, K, NF))
    for i, (seqs, yy, fs, _) in enumerate(batch):
        for j, s in enumerate(seqs):
            ids[i, j, :len(s)] = torch.tensor(s)
            am[i, j, :len(s)] = 1
            pm[i, j] = True
            y[i, j] = yy[j]
            F[i, j] = torch.tensor(fs[j])
    return ids, am, pm, y, F


class M(nn.Module):
    def __init__(self, base, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h + NF, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, ids, am, F):
        B, K, L = ids.shape
        o = self.enc(input_ids=ids.view(B * K, L),
                     attention_mask=am.view(B * K, L)).last_hidden_state[:, 0]
        x = torch.cat([o, F.view(B * K, NF)], dim=1)
        return self.head(self.drop(x)).view(B, K)


def evaluate(model, dl):
    model.eval()
    R = []
    with torch.no_grad():
        for ids, am, pm, y, F in dl:
            s = model(ids.cuda(), am.cuda(), F.cuda()).float().cpu().masked_fill(~pm, -1e4)
            pick = s.argmax(1)
            for i in range(s.shape[0]):
                R.append(float(y[i, pick[i]]))
    return R


def rule_and_ind(rows):
    IND, JR = [], []
    for r in rows:
        cs = r["cands"]
        ok = {(x[0], x[1]) for x in r["gold"]}
        IND.append(ok <= {(cs[0]["r"], cs[0]["c"]), (cs[1]["r"], cs[1]["c"])})
        u = np.array([max(1e-6, c["gscore"]) for c in cs])
        q = r["q"].lower()
        grow = bool(re.search(r"change|increase|decrease|growth|from .* to ", q))
        share = bool(re.search(r"percent of|percentage of|share of|portion|ratio of", q))
        best, bp = -1e9, None
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                sr, sc_ = a["r"] == b["r"], a["c"] == b["c"]
                bonus = 2.0 if (sr != sc_) else 0.2
                if grow and sr and not sc_:
                    bonus += 1.5
                if share and sc_ and not sr:
                    bonus += 1.5
                ra, rb = set(a.get("row_t") or []), set(b.get("row_t") or [])
                ca, cb = set(a.get("col_t") or []), set(b.get("col_t") or [])
                ov = (len(ra & rb) + len(ca & cb)) / max(1, len(ra | rb) + len(ca | cb))
                v = u[i] / u.max() + u[j] / u.max() + bonus - 0.5 * ov
                if v > best:
                    best, bp = v, {(a["r"], a["c"]), (b["r"], b["c"])}
        JR.append(ok <= bp)
    return IND, JR


def mcn(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    m = x + y
    if not m:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="r1c")
    ap.add_argument("--loss", default="bce", choices=["bce", "ce"])
    ap.add_argument("--suffix", default="", help="数据后缀（_full = 整表池，空 = 压缩池）")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = json.load(open("../results/r1_pairs_train%s.json" % args.suffix, encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev%s.json" % args.suffix, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    # 从训练集切一个验证集（按文档），用于选 epoch；dev 全程不参与任何选择
    docs = sorted({r["doc"] for r in tr})
    rng = random.Random(0)
    rng.shuffle(docs)
    val_docs = set(docs[:len(docs) // 10])
    tr2 = [r for r in tr if r["doc"] not in val_docs]
    va = [r for r in tr if r["doc"] in val_docs]
    print("训练 %d 题 / 验证 %d 题 / 评测 %d 题" % (len(tr2), len(va), len(dv)), flush=True)

    ds_tr = QDS(tr2, tok, train=True, seed=args.seed)
    ds_va = QDS(va, tok, train=False)
    ds_dv = QDS(dv, tok, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, pad))
    dl_va = DataLoader(ds_va, batch_size=16, shuffle=False, collate_fn=lambda b: collate(b, pad))
    dl_dv = DataLoader(ds_dv, batch_size=16, shuffle=False, collate_fn=lambda b: collate(b, pad))

    model = M(args.base).cuda()
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss()
    npos = sum(p[2] for it in ds_tr.items for p in it[1])
    ntot = sum(len(it[1]) for it in ds_tr.items)
    pw = torch.tensor([(ntot - npos) / max(1, npos)], device="cuda")
    lossf_bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    print("训练对 %d，正例 %d（%.1f%%），pos_weight=%.1f" % (ntot, npos, 100 * npos / ntot, pw.item()), flush=True)

    best = (-1, None, -1)
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for bi, (ids, am, pm, y, F) in enumerate(dl_tr):
            s = model(ids.cuda(), am.cuda(), F.cuda())
            pmm = pm.cuda()
            if args.loss == "ce":
                s = s.masked_fill(~pmm, -1e4)
                keep = y.sum(1) > 0
                if keep.sum() == 0:
                    continue
                tgt = y[keep].float().argmax(1)
                loss = lossf(s[keep.cuda()], tgt.cuda())
            else:
                keep = pmm
                loss = lossf_bce(s[keep], y.cuda()[keep])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            if bi % 40 == 0:
                print("  ep%d %d/%d loss %.4f" % (ep + 1, bi, len(dl_tr), tot / max(1, n)), flush=True)
        v = sum(evaluate(model, dl_va)) / len(ds_va)
        print("epoch %d：验证整对命中 %.4f" % (ep + 1, v), flush=True)
        if v > best[0]:
            best = (v, {k: t.cpu().clone() for k, t in model.state_dict().items()}, ep + 1)

    model.load_state_dict(best[1])
    print("取 epoch %d（验证 %.4f）" % (best[2], best[0]), flush=True)
    R = evaluate(model, dl_dv)
    IND, JR = rule_and_ind(dv)
    ORC = [any(p[2] for p in r["pairs"]) for r in dv]
    print("\n=== 评测（%d 题，dev 未参与选模）===" % len(dv))
    print("候选池上界 %.4f" % (sum(ORC) / len(ORC)))
    print("Ind              %.4f" % (sum(IND) / len(IND)))
    print("Joint-R（规则）  %.4f" % (sum(JR) / len(JR)))
    print("%s（语义+结构） %.4f" % (args.tag, sum(R) / len(R)))
    for name, base in (("vs Joint-R", JR), ("vs Ind", IND)):
        x, y, p = mcn(R, base)
        print("  %s %s：%d 对 / %d 错，p=%.4g" % (args.tag, name, x, y, p))
    torch.save({"base": args.base, "state": model.state_dict()}, "../results/%s.pt" % args.tag)


if __name__ == "__main__":
    main()
