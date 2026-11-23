# -*- coding: utf-8 -*-
"""同容量「独立打分」基线：双塔（bi-encoder），用来检验联合建模本身是否必要。

与 R1（cross-encoder，同时看两格）对照：
  - 同一个底座、同一份数据、同样轮数；
  - 只把「一次看两个格」换成「分别给每个格打分，取分数最高的两个」。

若双塔与 R1 打平，说明"联合"这件事没有价值，R1 的增益只是"换了个更好的打分器"；
若双塔明显更低，才说明收益来自建模两个格子的关系。
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
MAX_Q, MAX_C = 64, 32


def enc(tok, text, n):
    return tok(text, add_special_tokens=False)["input_ids"][:n]


class CellDS(Dataset):
    """一条样本 = (问题, 一个候选格)，标签 = 它是否金标操作数。"""

    def __init__(self, rows, tok):
        self.tok = tok
        self.items = []
        for r in rows:
            gold = set(map(tuple, r["gold"]))
            for c in r["cands"]:
                self.items.append((r["q"], c["txt"], 1.0 if (c["r"], c["c"]) in gold else 0.0,
                                   r["key"], c["r"], c["c"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        q, c, y, key, r, cc = self.items[k]
        t = self.tok
        qa = [t.cls_token_id] + enc(t, q, MAX_Q) + [t.sep_token_id]
        ca = [t.cls_token_id] + enc(t, c, MAX_C) + [t.sep_token_id]
        return qa, ca, y, k


def collate(batch, pad):
    B = len(batch)
    Lq = max(len(b[0]) for b in batch)
    Lc = max(len(b[1]) for b in batch)
    q = torch.full((B, Lq), pad, dtype=torch.long)
    qm = torch.zeros((B, Lq), dtype=torch.long)
    c = torch.full((B, Lc), pad, dtype=torch.long)
    cm = torch.zeros((B, Lc), dtype=torch.long)
    y = torch.zeros(B)
    ks = []
    for i, (qa, ca, yy, k) in enumerate(batch):
        q[i, :len(qa)] = torch.tensor(qa)
        qm[i, :len(qa)] = 1
        c[i, :len(ca)] = torch.tensor(ca)
        cm[i, :len(ca)] = 1
        y[i] = yy
        ks.append(k)
    return q, qm, c, cm, y, torch.tensor(ks)


class Bi(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.proj = nn.Linear(h, h)
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, q, qm, c, cm):
        qe = self.enc(input_ids=q, attention_mask=qm).last_hidden_state[:, 0]
        ce = self.enc(input_ids=c, attention_mask=cm).last_hidden_state[:, 0]
        qe = nn.functional.normalize(self.proj(qe), dim=-1)
        ce = nn.functional.normalize(ce, dim=-1)
        return (qe * ce).sum(-1) * self.temp


def select_top2(model, rows, tok):
    """每题给所有候选打分，取前二；返回逐题是否正确。"""
    model.eval()
    hits, picks = [], {}
    for r in rows:
        gold = set(map(tuple, r["gold"]))
        sc = []
        with torch.no_grad():
            for c in r["cands"]:
                qa = [tok.cls_token_id] + enc(tok, r["q"], MAX_Q) + [tok.sep_token_id]
                ca = [tok.cls_token_id] + enc(tok, c["txt"], MAX_C) + [tok.sep_token_id]
                s = model(torch.tensor([qa]).cuda(), torch.ones(1, len(qa), dtype=torch.long).cuda(),
                          torch.tensor([ca]).cuda(), torch.ones(1, len(ca), dtype=torch.long).cuda())
                sc.append(float(s[0]))
        order = np.argsort(-np.array(sc))[:2]
        top = {(r["cands"][i]["r"], r["cands"][i]["c"]) for i in order}
        hits.append(gold <= top)
        picks[r["qid"]] = [[r["cands"][i]["r"], r["cands"][i]["c"]] for i in order]
    return hits, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="bi")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = json.load(open("../results/r1_pairs_train_full.json", encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev_full.json", encoding="utf-8"))
    te = json.load(open("../results/xds_pairs_test.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id or 0

    ds = CellDS(tr, tok)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, pad))
    npos = sum(x[2] for x in ds.items)
    print("训练格 %d，正例 %d（%.1f%%）" % (len(ds), npos, 100 * npos / len(ds)), flush=True)

    model = Bi(args.base).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    pw = torch.tensor([(len(ds) - npos) / max(1, npos)], device="cuda")
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for q, qm, c, cm, y, _ in dl:
            s = model(q.cuda(), qm.cuda(), c.cuda(), cm.cuda())
            loss = lossf(s, y.cuda())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        hd, _ = select_top2(model, dv, tok)
        ht, _ = select_top2(model, te, tok)
        print("epoch %d：loss %.4f | dev 整对 %.3f | test 整对 %.3f"
              % (ep + 1, tot / n, sum(hd) / len(hd), sum(ht) / len(ht)), flush=True)

    hd, pd = select_top2(model, dv, tok)
    ht, pt = select_top2(model, te, tok)
    print("\n=== 同容量独立打分（双塔）===")
    print("dev  114 题：%.3f（R1 0.570）" % (sum(hd) / len(hd)))
    print("test 182 题：%.3f（R1 0.511）" % (sum(ht) / len(ht)))
    json.dump(pd, open("../results/bi_force_dev.json", "w"), ensure_ascii=False)
    json.dump(pt, open("../results/bi_force_test.json", "w"), ensure_ascii=False)
    torch.save({"base": args.base, "state": model.state_dict()}, "../results/%s.pt" % args.tag)


if __name__ == "__main__":
    main()
