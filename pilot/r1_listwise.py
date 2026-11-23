# -*- coding: utf-8 -*-
"""R1b：listwise 版成对选择器。

与 R1（逐对 BCE）的区别只在训练目标：这里每题一次 softmax，
直接在「该题的全部候选对」上做交叉熵，优化的正是推理时的
argmax 选择，而不是「每一对是否为正」这个代理目标。
"""
import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

MAX_Q = 64
MAX_C = 32


def encode(tok, q, a, b):
    qa = tok(q, add_special_tokens=False)["input_ids"][:MAX_Q]
    ca = tok(a, add_special_tokens=False)["input_ids"][:MAX_C]
    cb = tok(b, add_special_tokens=False)["input_ids"][:MAX_C]
    return [tok.cls_token_id] + qa + [tok.sep_token_id] + ca + [tok.sep_token_id] + cb + [tok.sep_token_id]


class QDS(Dataset):
    """一条样本 = 一道题（含它的全部候选对）。"""

    def __init__(self, rows, tok, train=False, max_pairs=16, seed=0):
        rng = random.Random(seed)
        self.tok = tok
        self.items = []
        for r in rows:
            ps = r["pairs"]
            if train and len(ps) > max_pairs:
                pos = [p for p in ps if p[2]]
                neg = [p for p in ps if not p[2]]
                k = max(0, max_pairs - len(pos))
                ps = pos + rng.sample(neg, min(len(neg), k))
            if train and not any(p[2] for p in ps):
                continue
            self.items.append((r, ps))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        r, ps = self.items[k]
        seqs = [encode(self.tok, r["q"], r["cands"][i]["txt"], r["cands"][j]["txt"])
                for i, j, _ in ps]
        y = [p[2] for p in ps]
        return seqs, y, r["key"]


def collate(batch, pad_id):
    B = len(batch)
    K = max(len(b[0]) for b in batch)
    L = max(len(s) for b in batch for s in b[0])
    ids = torch.full((B, K, L), pad_id, dtype=torch.long)
    am = torch.zeros((B, K, L), dtype=torch.long)
    pm = torch.zeros((B, K), dtype=torch.bool)      # 该位置有候选对
    y = torch.zeros((B, K), dtype=torch.float)
    for i, (seqs, yy, _) in enumerate(batch):
        for j, s in enumerate(seqs):
            ids[i, j, :len(s)] = torch.tensor(s)
            am[i, j, :len(s)] = 1
            pm[i, j] = True
            y[i, j] = yy[j]
    return ids, am, pm, y


class CE(nn.Module):
    def __init__(self, base, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, 1)

    def forward(self, ids, am):
        B, K, L = ids.shape
        o = self.enc(input_ids=ids.view(B * K, L),
                     attention_mask=am.view(B * K, L)).last_hidden_state[:, 0]
        return self.head(self.drop(o)).view(B, K)


def listwise_loss(logits, pm, y, temperature=1.0):
    logits = logits / temperature
    logits = logits.masked_fill(~pm, -1e4)
    # 正例可能不止一个：对每个正例算 CE 再平均
    n_pos = y.sum(1)
    ok = n_pos > 0
    if ok.sum() == 0:
        return None
    logits, y2 = logits[ok], y[ok]
    logp = torch.log_softmax(logits, dim=1)
    num = (logp * y2).sum(1)
    return -(num / y2.sum(1).clamp(min=1)).mean()


def eval_set(model, dl):
    model.eval()
    hits, n, scores_all, keys_all = 0, 0, [], []
    with torch.no_grad():
        for ids, am, pm, y in dl:
            s = model(ids.cuda(), am.cuda()).float().cpu()
            s = s.masked_fill(~pm, -1e4)
            pick = s.argmax(1)
            for i in range(s.shape[0]):
                hits += float(y[i, pick[i]])
                n += 1
            scores_all.append(s)
            keys_all.append(pm)
    return hits / max(1, n), torch.cat(scores_all)


def pair_auc(model, dl):
    model.eval()
    sc, yy = [], []
    with torch.no_grad():
        for ids, am, pm, y in dl:
            s = model(ids.cuda(), am.cuda()).float().cpu()
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    if pm[i, j]:
                        sc.append(float(s[i, j]))
                        yy.append(float(y[i, j]))
    sc, yy = np.array(sc), np.array(yy)
    order = np.argsort(sc)
    ranks = np.empty(len(sc))
    ranks[order] = np.arange(1, len(sc) + 1)
    npos, nneg = int(yy.sum()), int((1 - yy).sum())
    return (ranks[yy == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg)


def rule_and_ind(rows):
    import re
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)      # 每批题数
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="r1b")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = json.load(open("../results/r1_pairs_train.json", encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    ds_tr = QDS(tr, tok, train=True, seed=args.seed)
    ds_dv = QDS(dv, tok, train=False)
    print("训练题 %d / 评测题 %d" % (len(ds_tr), len(ds_dv)), flush=True)

    from torch.utils.data import DataLoader
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       collate_fn=lambda b: collate(b, pad))
    dl_dv = DataLoader(ds_dv, batch_size=16, shuffle=False,
                       collate_fn=lambda b: collate(b, pad))

    model = CE(args.base).cuda()
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": list(model.head.parameters()) + list(model.drop.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.15)

    best = (-1, None)
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for bi, (ids, am, pm, y) in enumerate(dl_tr):
            logits = model(ids.cuda(), am.cuda())
            loss = listwise_loss(logits, pm.cuda(), y.cuda())
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            if bi % 40 == 0:
                print("  ep%d %d/%d loss %.4f" % (ep + 1, bi, len(dl_tr), tot / max(1, n)), flush=True)
        hit, _ = eval_set(model, dl_dv)
        auc = pair_auc(model, dl_dv)
        print("epoch %d：listwise 整对命中 %.4f | pair AUC %.4f" % (ep + 1, hit, auc), flush=True)
        if hit > best[0]:
            best = (hit, {k: v.cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    hit, _ = eval_set(model, dl_dv)
    IND, JR = rule_and_ind(dv)
    ORC = [any(p[2] for p in r["pairs"]) for r in dv]

    def mcn(a, b):
        x = sum(1 for p, q in zip(a, b) if p and not q)
        y = sum(1 for p, q in zip(a, b) if q and not p)
        m = x + y
        if not m:
            return x, y, 1.0
        k = min(x, y)
        return x, y, min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)

    # 逐题正确向量（用于配对检验）
    model.eval()
    R1 = []
    with torch.no_grad():
        for ids, am, pm, y in dl_dv:
            s = model(ids.cuda(), am.cuda()).float().cpu().masked_fill(~pm, -1e4)
            pick = s.argmax(1)
            for i in range(s.shape[0]):
                R1.append(float(y[i, pick[i]]))

    print("\n=== 评测（%d 题）===" % len(dv))
    print("候选池上界 %.4f" % (sum(ORC) / len(ORC)))
    print("Ind        %.4f" % (sum(IND) / len(IND)))
    print("Joint-R    %.4f" % (sum(JR) / len(JR)))
    print("%s       %.4f（最佳 epoch）" % (args.tag, hit))
    for name, base in (("vs Joint-R", JR), ("vs Ind", IND)):
        x, y, p = mcn(R1, base)
        print("  %s %s：%d 对 / %d 错，p=%.4g" % (args.tag, name, x, y, p))
    torch.save({"base": args.base, "state": model.state_dict(),
                "hit": hit}, "../results/%s.pt" % args.tag)


if __name__ == "__main__":
    main()
