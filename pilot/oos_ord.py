# -*- coding: utf-8 -*-
"""OOS：操作数–算子选择器 + 精确执行（预分词版）。

动机：压缩到 3–20% 后，端到端错误约七成落在"算式层"，而不是"找错数"。
把表达式构造从 3B 模型手里拿走：小模块在候选格上选一对 + 判一个算式，
答案由脚本精确求值。16% 的"输出不可解析"与 70% 的"算错"因此同时消失，
天花板 = P(选对一对) × P(判对算式)。

标签不解析 FinQA 的 op 字符串：拿 gold 两格的值试 8 个算式，
看哪个能还原 gold 答案；对不上的题（多步）不进算子训练，但计入评测。
"""
import argparse
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from ground import tokens
from r1_feat import MAX_C, MAX_Q, pair_features as _pf
NF = 13   # 11 维结构特征 + 2 维「提及顺序」


def _mpos(c, q):
    low = q.lower()
    best = None
    for nm in (c.get("row", ""), c.get("col", "")):
        for t in tokens(nm):
            if len(t) < 3:
                continue
            k = low.find(t.lower())
            if k >= 0 and (best is None or k < best):
                best = k
    return best


def pair_features(cands, i, j, q, gmax, qw):
    f = _pf(cands, i, j, q, gmax, qw)
    pi, pj = _mpos(cands[i], q), _mpos(cands[j], q)
    if pi is None or pj is None:
        flag, gap = 0.5, 0.0
    else:
        flag = 1.0 if pi <= pj else 0.0
        gap = min(abs(pi - pj) / 50.0, 1.0)
    return f + [flag, gap]

FORMULAS = [
    ("a+b", lambda a, b: a + b),
    ("a-b", lambda a, b: a - b),
    ("b-a", lambda a, b: b - a),
    ("a*b", lambda a, b: a * b),
    ("a/b", lambda a, b: a / b if b else None),
    ("b/a", lambda a, b: b / a if a else None),
    ("(a-b)/b", lambda a, b: (a - b) / b if b else None),
    ("(b-a)/a", lambda a, b: (b - a) / a if a else None),
]
NOP = len(FORMULAS)
TOL = 1e-3


def close(x, y):
    return x is not None and abs(x - y) <= TOL * max(1.0, abs(y))


def gold_pair(r):
    for i, j, lab in r["pairs"]:
        if lab:
            return i, j
    return None


def opp_of(r):
    gi = gold_pair(r)
    if gi is None:
        return -1
    a, b = r["cands"][gi[0]]["v"], r["cands"][gi[1]]["v"]
    y = r["key"][1]
    for k, (_, f) in enumerate(FORMULAS):
        try:
            if close(f(a, b), y):
                return k
        except ZeroDivisionError:
            continue
    return -1


class DS(Dataset):
    def __init__(self, rows, tok, train=False, seed=0, max_pos_neg=16, cap_eval=32):
        self.tok = tok
        rng = random.Random(seed)
        self.items = []
        for r in rows:
            ps = r["pairs"]
            if len(ps) > cap_eval:
                pos = [p for p in ps if p[2]]
                neg = sorted([p for p in ps if not p[2]],
                             key=lambda p: -(r["cands"][p[0]]["gscore"] +
                                             r["cands"][p[1]]["gscore"]))
                ps = pos + neg[:max(0, cap_eval - len(pos))]
            if train:
                pos = [p for p in ps if p[2]]
                if not pos:
                    continue
                if len(ps) > max_pos_neg:
                    ps = pos + rng.sample([p for p in ps if not p[2]],
                                          min(len(ps) - len(pos), max_pos_neg - len(pos)))
            elif len(ps) > cap_eval:
                ps = ps[:cap_eval]
            gmax = max(c["gscore"] for c in r["cands"]) or 1.0
            qw = set(tokens(r["q"]))
            qa = tok(r["q"], add_special_tokens=False)["input_ids"][:MAX_Q]
            seqs, ys, fs, ijs = [], [], [], []
            for i, j, lab in ps:
                ca = tok(r["cands"][i]["txt"], add_special_tokens=False)["input_ids"][:MAX_C]
                cb = tok(r["cands"][j]["txt"], add_special_tokens=False)["input_ids"][:MAX_C]
                seqs.append([tok.cls_token_id] + qa + [tok.sep_token_id] + ca +
                            [tok.sep_token_id] + cb + [tok.sep_token_id])
                ys.append(float(lab))
                fs.append(pair_features(r["cands"], i, j, r["q"], gmax, qw))
                ijs.append((i, j))
            self.items.append(dict(key=tuple(r["key"]), seqs=seqs, y=ys, f=fs,
                                   op=opp_of(r), ij=ijs, row=r))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        it = self.items[k]
        return it["key"], it["seqs"], it["y"], it["f"], it["op"], it["ij"], k


def coll(batch, pad):
    B = len(batch)
    K = max(len(b[1]) for b in batch)
    L = max(len(s) for b in batch for s in b[1])
    ids = torch.full((B, K, L), pad, dtype=torch.long)
    am = torch.zeros((B, K, L), dtype=torch.long)
    pm = torch.zeros((B, K), dtype=torch.bool)
    y = torch.zeros((B, K))
    F = torch.zeros((B, K, NF))
    op = torch.tensor([b[4] for b in batch], dtype=torch.long)
    for i, (_, seqs, ys, fs, _, _, _) in enumerate(batch):
        for j, s in enumerate(seqs):
            ids[i, j, :len(s)] = torch.tensor(s)
            am[i, j, :len(s)] = 1
            pm[i, j] = True
            y[i, j] = ys[j]
            F[i, j] = torch.tensor(fs[j])
    return ids, am, pm, y, F, op, batch


class M2(nn.Module):
    def __init__(self, base, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.pair = nn.Sequential(nn.Linear(h + NF, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 1))
        self.op = nn.Sequential(nn.Linear(h + NF, 256), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(256, NOP))

    def forward(self, ids, am, F):
        B, K, L = ids.shape
        o = self.enc(input_ids=ids.view(B * K, L),
                     attention_mask=am.view(B * K, L)).last_hidden_state[:, 0]
        x = self.drop(torch.cat([o, F.view(B * K, NF)], dim=1))
        return self.pair(x).view(B, K), self.op(x).view(B, K, NOP)


def run_epoch(model, dl, opt=None, sched=None, op_w=0.5):
    train = opt is not None
    model.train(train)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    tot, n = 0.0, 0
    for ids, am, pm, y, F, op, _ in dl:
        ids, am, pm, y, F, op = (x.cuda() for x in (ids, am, pm, y, F, op))
        s, o = model(ids, am, F)
        has = y.sum(1) > 0
        sel = y.float().argmax(1)
        loss = ce(s.masked_fill(~pm, -1e4), sel) if has.any() else s.sum() * 0
        op_t = torch.full_like(op, -1)
        op_t[has] = op[has]
        if has.any() and (op_t >= 0).any():
            op_sel = o[torch.arange(o.shape[0], device=o.device), sel]
            loss = loss + op_w * ce(op_sel, op_t)
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
        tot += float(loss.detach())
        n += 1
        if train and n % 20 == 0:
            print("   batch %d/%d  loss %.4f" % (n, len(dl), tot / n), flush=True)
    return tot / max(1, n)


@torch.no_grad()
def evaluate(model, ds, dl, dump=None):
    model.eval()
    st = dict(n=0, pair=0, op_n=0, op_ok=0, em=0, em_goldop=0, em_goldpair=0,
              n_lab=0, em_lab=0)
    for ids, am, pm, y, F, op, batch in dl:
        s, o = model(ids.cuda(), am.cuda(), F.cuda())
        s = s.float().cpu().masked_fill(~pm, -1e4)
        o = o.float().cpu()
        for b in range(s.shape[0]):
            idx = batch[b][6]
            it = ds.items[idx]
            r = it["row"]
            st["n"] += 1
            pick = int(s[b].argmax())
            if not bool(pm[b, pick]):
                continue
            i, j = it["ij"][pick]
            gi = gold_pair(r)
            ok_pair = gi is not None and {i, j} == set(gi)
            st["pair"] += ok_pair
            k = int(o[b, pick].argmax())
            a, bb = r["cands"][i]["v"], r["cands"][j]["v"]
            yt = r["key"][1]
            go = it["op"]

            def ev(opk):
                try:
                    return close(FORMULAS[opk][1](a, bb), yt)
                except ZeroDivisionError:
                    return False

            ok = ev(k)
            st["em"] += ok
            if go >= 0:
                st["n_lab"] += 1
                st["em_lab"] += ok
                st["em_goldop"] += ev(go)
                if ok_pair:
                    st["op_n"] += 1
                    st["op_ok"] += (k == go)
                    st["em_goldpair"] += ev(go)
            if dump is not None:
                dump.append(dict(key=list(it["key"]), pair=[i, j], gold=gi,
                                 op=k, gold_op=go, ok=bool(ok),
                                 ok_pair=bool(ok_pair)))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="E:/KV_models/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bs_eval", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--train_suffix", default=None)
    ap.add_argument("--tag", default="oos")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tsuf = args.suffix if args.train_suffix is None else args.train_suffix
    tr = json.load(open("../results/r1_pairs_train%s.json" % tsuf, encoding="utf-8"))
    dv = json.load(open("../results/r1_pairs_dev%s.json" % args.suffix, encoding="utf-8"))
    ltr = sum(1 for r in tr if opp_of(r) >= 0)
    ldv = sum(1 for r in dv if opp_of(r) >= 0)
    print("训练 %d 题，%d 题有算式标签（%.1f%%）｜ dev %d 题，%d 题有标签（%.1f%%）"
          % (len(tr), ltr, 100 * ltr / len(tr), len(dv), ldv, 100 * ldv / len(dv)),
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    docs = sorted({r["doc"] for r in tr})
    rng = random.Random(0)
    rng.shuffle(docs)
    vd = set(docs[:max(1, len(docs) // 10)])
    tr2 = [r for r in tr if r["doc"] not in vd]
    va = [r for r in tr if r["doc"] in vd]

    ds_tr = DS(tr2, tok, train=True, seed=args.seed)
    ds_va = DS(va, tok)
    ds_dv = DS(dv, tok)
    print("样本：训练 %d / 验证 %d / dev %d；平均候选对 %.1f"
          % (len(ds_tr), len(ds_va), len(ds_dv),
             sum(len(i["ij"]) for i in ds_tr.items) / len(ds_tr)), flush=True)

    f = lambda b: coll(b, pad)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=f)
    dl_va = DataLoader(ds_va, batch_size=args.bs_eval, shuffle=False, collate_fn=f)
    dl_dv = DataLoader(ds_dv, batch_size=args.bs_eval, shuffle=False, collate_fn=f)

    model = M2(args.base).cuda()
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": model.pair.parameters(), "lr": args.head_lr},
        {"params": model.op.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    steps = max(1, len(dl_tr) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr, args.head_lr],
        total_steps=steps, pct_start=0.15)

    best = (-1.0, None, -1)
    for ep in range(args.epochs):
        print("  [ep%d] 训练开始" % (ep+1), flush=True)
        loss = run_epoch(model, dl_tr, opt, sched)
        print("  [ep%d] 训练结束，验证中" % (ep+1), flush=True)
        st = evaluate(model, ds_va, dl_va)
        n = st["n"]
        print("ep%d loss %.4f | val 成对 %.3f 算式 %.3f EM %.3f"
              % (ep + 1, loss, st["pair"] / n, st["op_ok"] / max(1, st["op_n"]),
                 st["em"] / n), flush=True)
        if st["em"] > best[0]:
            best = (st["em"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, ep + 1)

    model.load_state_dict(best[1])
    dump = []
    st = evaluate(model, ds_dv, dl_dv, dump=dump)
    n = st["n"]
    print("\n=== dev（%d 题，取 epoch %d）===" % (n, best[2]))
    print("成对命中                %.4f（%d 题）" % (st["pair"] / n, st["pair"]))
    print("算式正确（条件在成对上）  %.4f（%d/%d）"
          % (st["op_ok"] / max(1, st["op_n"]), st["op_ok"], st["op_n"]))
    print("端到端 EM（全部 %d 题）  %.4f" % (n, st["em"] / n))
    print("端到端 EM（单步 %d 题）  %.4f   ← 与 3B 的 0.373 可比"
          % (st["n_lab"], st["em_lab"] / max(1, st["n_lab"])))
    print("上界 A：配对正确 + gold 算式 %.4f（全部题）" % (st["em_goldpair"] / n))
    print("上界 B：任意配对 + gold 算式 %.4f（全部题）" % (st["em_goldop"] / n))
    print("对照：3B 端到端 单步 3%%=0.373 / 20%%=0.410（含 16%% 输出不可解析）")
    json.dump(dump, open("../results/%s_pred.json" % args.tag, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    torch.save({"base": args.base, "state": model.state_dict(),
                "formulas": [x[0] for x in FORMULAS]}, "../results/%s.pt" % args.tag)


if __name__ == "__main__":
    main()
