# -*- coding: utf-8 -*-
"""把「成对交叉编码器」的收益拆开：问题×单元格交互 vs 两格联合建模。

三档同底座、同数据、同轮数的文本模型（都不加结构特征）：
  twin   : 双塔，q 与 c 各自编码后点积            —— 两者都没有（既有 r1_bi）
  cell   : 单格交叉编码器 [CLS] q [SEP] c [SEP]    —— 有 q-c 交互，无联合
  pair   : 成对交叉编码器 [CLS] q [SEP] ci [SEP] cj —— 两者都有
部署版（r1_feat，交叉编码器 + 11 维结构特征）另作第四档，用来量"结构特征"的贡献。

用法：
  python r1_cell.py --mode cell --tag ce_cell
  python r1_cell.py --mode pair --tag ce_pair
"""
import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MAX_Q, MAX_C = 64, 32


def make_ids(tok, q, texts):
    ids = [tok.cls_token_id] + tok(q, add_special_tokens=False)["input_ids"][:MAX_Q] + [tok.sep_token_id]
    for t in texts:
        ids += tok(t, add_special_tokens=False)["input_ids"][:MAX_C] + [tok.sep_token_id]
    return ids


class DS(Dataset):
    def __init__(self, rows, tok, mode, feat=False):
        self.tok = tok
        self.items = []
        if feat:
            from r1_feat import pair_features
            from ground import tokens
        for r in rows:
            gold = set(map(tuple, r["gold"]))
            if mode == "cell":
                for c in r["cands"]:
                    self.items.append((r["q"], [c["txt"]],
                                       1.0 if (c["r"], c["c"]) in gold else 0.0,
                                       [0.0] * 11))
            else:
                gmax = max(c["gscore"] for c in r["cands"]) or 1.0
                qw = set(tokens(r["q"])) if feat else None
                for i, j, y in r["pairs"]:
                    f = pair_features(r["cands"], i, j, r["q"], gmax, qw) if feat else [0.0] * 11
                    self.items.append((r["q"], [r["cands"][i]["txt"], r["cands"][j]["txt"]],
                                       float(y), f))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        q, texts, y, f = self.items[k]
        return make_ids(self.tok, q, texts), y, f


def collate(batch, pad):
    L = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    y = torch.zeros(len(batch))
    F = torch.zeros((len(batch), 11))
    for i, (sq, yy, f) in enumerate(batch):
        ids[i, :len(sq)] = torch.tensor(sq)
        am[i, :len(sq)] = 1
        y[i] = yy
        F[i] = torch.tensor(f)
    return ids, am, y, F


class M(nn.Module):
    def __init__(self, base, dropout=0.1, nf=11):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.nf = nf
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h + nf, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, ids, am, F):
        o = self.enc(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        x = torch.cat([o, F], dim=1) if self.nf else o
        return self.head(self.drop(x)).view(-1)


def _logit(model, tok, q, texts, f=None):
    ids = torch.tensor([make_ids(tok, q, texts)]).cuda()
    F = torch.tensor([f if f else [0.0] * 11], dtype=torch.float).cuda()
    with torch.no_grad():
        return float(model(ids, torch.ones_like(ids), F)[0])


def score_rows(model, rows, tok, mode):
    """逐题选出前二格，返回「整对命中」布尔列表。"""
    if mode == "pair":
        from r1_feat import pair_features
        from ground import tokens
    model.eval()
    hits = []
    for r in rows:
        gold = set(map(tuple, r["gold"]))
        if mode == "cell":
            sc = [_logit(model, tok, r["q"], [c["txt"]]) for c in r["cands"]]
            order = np.argsort(-np.array(sc))[:2]
            pick = {(r["cands"][i]["r"], r["cands"][i]["c"]) for i in order}
        else:
            gmax = max(c["gscore"] for c in r["cands"]) or 1.0
            qw = set(tokens(r["q"]))
            sc = [_logit(model, tok, r["q"],
                         [r["cands"][i]["txt"], r["cands"][j]["txt"]],
                         pair_features(r["cands"], i, j, r["q"], gmax, qw))
                  for i, j, _ in r["pairs"]]
            i, j, _ = r["pairs"][int(np.argmax(sc))]
            pick = {(r["cands"][i]["r"], r["cands"][i]["c"]),
                    (r["cands"][j]["r"], r["cands"][j]["c"])}
        hits.append(gold <= pick)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--mode", default="cell", choices=["cell", "pair"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ce_cell")
    ap.add_argument("--feat", action="store_true", help="成对模式附加 11 维结构特征")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = json.load(open("../results/r1_pairs_train_full.json", encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev_full.json", encoding="utf-8"))
    te = json.load(open("../results/xds_pairs_test.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    docs = sorted({r["doc"] for r in tr})
    random.Random(0).shuffle(docs)
    val = set(docs[:len(docs) // 10])
    tr2 = [r for r in tr if r["doc"] not in val]
    va = [r for r in tr if r["doc"] in val]
    print("训练 %d 题 / 验证 %d 题 / dev %d / test %d" % (len(tr2), len(va), len(dv), len(te)), flush=True)

    ds = DS(tr2, tok, args.mode, feat=(args.mode == "pair" and args.feat))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, pad))
    npos = sum(it[2] for it in ds.items)
    pw = torch.tensor([(len(ds) - npos) / max(1, npos)], device="cuda")
    print("训练样本 %d，正例 %d（%.1f%%），pos_weight=%.1f"
          % (len(ds), npos, 100 * npos / len(ds), pw.item()), flush=True)

    model = M(args.base).cuda()
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=len(dl) * args.epochs,
        pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    best = (-1, None, -1)
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for bi, (ids, am, y, F) in enumerate(dl):
            s = model(ids.cuda(), am.cuda(), F.cuda())
            loss = lossf(s, y.cuda())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            if bi % 100 == 0:
                print("  ep%d %d/%d loss %.4f" % (ep + 1, bi, len(dl), tot / max(1, n)), flush=True)
        v = np.mean(score_rows(model, va, tok, args.mode))
        print("epoch %d：验证整对命中 %.4f" % (ep + 1, v), flush=True)
        if v > best[0]:
            best = (v, {k: t.cpu().clone() for k, t in model.state_dict().items()}, ep + 1)

    model.load_state_dict(best[1])
    print("取 epoch %d（验证 %.4f）" % (best[2], best[0]), flush=True)
    for name, rows in (("dev", dv), ("test", te)):
        h = score_rows(model, rows, tok, args.mode)
        print("%s：整对命中 %.4f（%d 题）" % (name, np.mean(h), len(h)), flush=True)
    torch.save({"base": args.base, "mode": args.mode, "state": model.state_dict()},
               "../results/%s.pt" % args.tag)


if __name__ == "__main__":
    main()
