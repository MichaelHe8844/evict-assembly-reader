# -*- coding: utf-8 -*-
"""R1：结构感知的成对打分器（cross-encoder）。

输入  [CLS] 问题 [SEP] 单元格A [SEP] 单元格B [SEP]
输出  logit：这一对能否完成该题的运算

与 C 的逻辑回归的区别：C 只用 13 个手工特征（两项单元格分、同行/同列、
数值比、表头重叠……），模型容量被特征上限锁死；这里让编码器直接读
「问题 + 两个格子的行名/列名/单位/数值」，自己学该看什么。

训练集：FinQA train 的单步题（与评测的 dev 文档完全不重叠）。
评测集：dev 上 126 题的子集（能建出 ≥2 候选的 114 题）。
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

MAX_Q = 64
MAX_C = 32


class PairDS(Dataset):
    def __init__(self, rows, tok, train=False, neg_per_q=12, swap=True, seed=0):
        self.items = []
        rng = random.Random(seed)
        for r in rows:
            pos = [p for p in r["pairs"] if p[2]]
            neg = [p for p in r["pairs"] if not p[2]]
            if train:
                neg = rng.sample(neg, min(len(neg), neg_per_q))
            for i, j, y in pos + neg:
                self.items.append((r["q"], r["cands"][i]["txt"], r["cands"][j]["txt"], y, r["key"]))
                if train and swap:
                    self.items.append((r["q"], r["cands"][j]["txt"], r["cands"][i]["txt"], y, r["key"]))
        self.tok = tok

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        q, a, b, y, _ = self.items[k]
        t = self.tok
        qa = t(q, add_special_tokens=False)["input_ids"][:MAX_Q]
        ca = t(a, add_special_tokens=False)["input_ids"][:MAX_C]
        cb = t(b, add_special_tokens=False)["input_ids"][:MAX_C]
        ids = [t.cls_token_id] + qa + [t.sep_token_id] + ca + [t.sep_token_id] + cb + [t.sep_token_id]
        return ids, float(y), k


def collate(batch, pad_id):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, _, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        am[i, :len(x)] = 1
    y = torch.tensor([y for _, y, _ in batch], dtype=torch.float)
    idx = torch.tensor([k for _, _, k in batch], dtype=torch.long)
    return ids, am, y, idx


class CrossEncoder(nn.Module):
    def __init__(self, base, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, 1)

    def forward(self, ids, am):
        o = self.enc(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        return self.head(self.drop(o)).squeeze(-1)


def pair_auc(model, loader, dev):
    model.eval()
    scores = []
    with torch.no_grad():
        for ids, am, y, idx in loader:
            s = model(ids.cuda(), am.cuda()).float().cpu()
            scores.append(s)
    s = torch.cat(scores).numpy()
    y = np.array([it[3] for it in dev.items])
    # AUC（Mann-Whitney）
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg)
    return auc, s


def select_metrics(rows, scores, dev_items):
    """把 pair 分数还原成「每题选一对」，算整对命中率。"""
    by_key = {}
    for k, (it, sc) in enumerate(zip(dev_items, scores)):
        q, a, b, y, key = it
        by_key.setdefault(tuple(key), []).append((sc, y))
    hit = 0
    by_q = {tuple(r["key"]): r for r in rows}
    for key, lst in by_key.items():
        r = by_q[key]
        # 每对在 swap 数据里出现两次，去重后取每题最高分的那一对
        best = max(lst, key=lambda x: x[0])
        hit += best[1]
    return hit / len(by_key)


def rule_and_ind(rows):
    """Ind（按 ground 分数取前二）与 Joint-R（手写规则）在同一批题上的基线。"""
    def same(a, b):
        return (a["r"] == b["r"]), (a["c"] == b["c"])

    def overlap(a, b):
        ra, rb = set(a["row_t"] or []), set(b["row_t"] or [])
        ca, cb = set(a["col_t"] or []), set(b["col_t"] or [])
        return (len(ra & rb) + len(ca & cb)) / max(1, len(ra | rb) + len(ca | cb))

    IND, JR = [], []
    for r in rows:
        cs = r["cands"]
        gold = {(c["r"], c["c"]) for c in cs}
        ok = {(x[0], x[1]) for x in r["gold"]}
        # Ind：候选排序的前两个
        pair = {(cs[0]["r"], cs[0]["c"]), (cs[1]["r"], cs[1]["c"])}
        IND.append(ok <= pair)
        # Joint-R：与 pair_select 的 sel_joint_rule 同一公式（不含向量项）
        u = np.array([max(1e-6, c.get("gscore", 1.0)) for c in cs])
        q = r["q"].lower()
        import re
        grow = bool(re.search(r"change|increase|decrease|growth|from .* to ", q))
        share = bool(re.search(r"percent of|percentage of|share of|portion|ratio of", q))
        best, bp = -1e9, None
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                sr, sc = same(a, b)
                bonus = 2.0 if (sr != sc) else 0.2
                if grow and sr and not sc:
                    bonus += 1.5
                if share and sc and not sr:
                    bonus += 1.5
                v = (u[i] / u.max()) + (u[j] / u.max()) + bonus - 0.5 * overlap(a, b)
                if v > best:
                    best, bp = v, {(a["r"], a["c"]), (b["r"], b["c"])}
        JR.append(ok <= bp)
    return IND, JR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = json.load(open("../results/r1_pairs_train.json", encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev.json", encoding="utf-8"))
    print("训练 %d 题 / 评测 %d 题" % (len(tr), len(dv)), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    assert all("gscore" in c for r in dv for c in r["cands"]), "缺 gscore，请重跑 r1_data.py"

    ds_tr = PairDS(tr, tok, train=True, neg_per_q=12, swap=True, seed=args.seed)
    ds_dv = PairDS(dv, tok, train=False, swap=False)
    print("训练对 %d（正例 %d）/ 评测对 %d" % (
        len(ds_tr), sum(x[3] for x in ds_tr.items), len(ds_dv)), flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0,
                       collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    dl_dv = DataLoader(ds_dv, batch_size=128, shuffle=False, num_workers=0,
                       collate_fn=lambda b: collate(b, tok.pad_token_id or 0))

    model = CrossEncoder(args.base).cuda()
    pos = sum(x[3] for x in ds_tr.items)
    neg = len(ds_tr) - pos
    pos_w = torch.tensor([neg / max(1, pos)], device="cuda")
    print("正负比 1:%.1f，pos_weight=%.1f" % (neg / max(1, pos), pos_w.item()), flush=True)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": list(model.head.parameters()) + list(model.drop.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.1)

    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for bi, (ids, am, y, _) in enumerate(dl_tr):
            ids, am, y = ids.cuda(), am.cuda(), y.cuda()
            loss = lossf(model(ids, am), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            if bi % 100 == 0:
                print("  ep%d %d/%d loss %.4f" % (ep + 1, bi, len(dl_tr), tot / n), flush=True)
        auc, scores = pair_auc(model, dl_dv, ds_dv)
        hit = select_metrics(dv, scores, ds_dv.items)
        print("epoch %d：pair AUC %.4f | 整对命中 %.4f" % (ep + 1, auc, hit), flush=True)

    IND, JR = rule_and_ind(dv)
    # 把 R1 的逐题结果取出来（每题取最高分那对）
    by_key = {}
    for it, sc in zip(ds_dv.items, scores):
        by_key.setdefault(tuple(it[4]), []).append((sc, it[3]))
    keys = [tuple(r["key"]) for r in dv]
    R1 = [max(by_key[k], key=lambda x: x[0])[1] for k in keys]
    ORC = [any(p[2] for p in r["pairs"]) for r in dv]   # 候选池里有正例 = 上界

    def mcn(a, b):
        x = sum(1 for p, q in zip(a, b) if p and not q)
        y = sum(1 for p, q in zip(a, b) if q and not p)
        m = x + y
        if not m:
            return x, y, 1.0
        k = min(x, y)
        return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)

    n_gold2 = sum(1 for r in dv if len(r["gold"]) >= 2)
    print("\n=== 评测（%d 题，其中两操作数 %d 题）===" % (len(dv), n_gold2))
    print("候选池上界（有正例即可选对） %.4f" % (sum(ORC) / len(ORC)))
    print("Ind（候选前二）     整对命中 %.4f" % (sum(IND) / len(IND)))
    print("Joint-R（手写规则） 整对命中 %.4f" % (sum(JR) / len(JR)))
    print("R1（交叉编码器）    整对命中 %.4f" % (sum(R1) / len(R1)))
    for name, base in (("vs Joint-R", JR), ("vs Ind", IND)):
        x, y, p = mcn(R1, base)
        print("  R1 %s：R1 对/基线错 %d，基线对/R1 错 %d，McNemar p=%.4g" % (name, x, y, p))

    torch.save({"base": args.base, "state": model.state_dict()}, "../results/r1_ce.pt")


if __name__ == "__main__":
    main()
