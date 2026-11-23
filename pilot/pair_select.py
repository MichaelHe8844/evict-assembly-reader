# -*- coding: utf-8 -*-
"""C 方案：对「单元格对」做联合打分，与「独立打分取前二」对照。

候选池完全相同（保留集内的单元格，grounder 取 top-8），三种选择器：
  Ind     单元格独立打分取前二
  Joint   s_θ(q, c_i, c_j) 联合打分，取 argmax
  Oracle  直接给 gold 对（上界）

训练集与评测集按 id 切开（评测集是 ground_keys 的 83 个问题，训练集取调参段中不在其中的样本）。
"""
import json
import math
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract, table_index  # noqa: E402
from arm_c import fit_budget  # noqa: E402
from ground import cells_from_segs, rank, tokens  # noqa: E402
from operand_loc import gold_cells  # noqa: E402
from emb_util import EmbScorer  # noqa: E402

EVAL_KEYS = {(k[0], round(float(k[1]), 6)) for k in json.load(open("../results/ground_keys.json"))}
RAW = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
S = EmbScorer("E:/KV_models/bge-small-en-v1.5")


def build(d, tok):
    """返回 dict(pool, gold, feats) 或 None。

    pool: [(score, cell)]，按 grounder 分数降序，最多 8 个
    gold: [(r,c)]，来自 gold 程序实参
    """
    doc, spans = tokenize_sample(tok, d["segs"])
    if len(doc) < 64 or len(doc) > 1200:
        return None
    mc = max(1, int((0.85 * int(0.20 * len(doc))) / 6))
    idx, _ = extract(d["segs"], d["question"], max_cells=mc)
    B = max(8, int(0.20 * len(doc)))
    keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], d["segs"], spans, B))
    cand = rank(cells_from_segs(d["segs"], spans, keep), d["question"], topk=8)
    if len(cand) < 2:
        return None
    gc = gold_cells(RAW.get(d["id"], {}))
    if not gc or any(x is None for x in gc):
        return None
    return doc, cand, [(c["r"], c["c"]) for c in gc if c]


def feats(cand, gold, q, qvec, qw):
    u = np.array([s for s, _ in cand], dtype=np.float32)
    u = u / max(1e-6, u.max())
    hv = [S.header((c["row"] + " " + c["col"]) if c["row"] or c["col"] else str(c["v"])) for _, c in cand]
    cs = [float(max(0.0, qvec @ v)) for v in hv]
    rows = [set(tokens(c["row"])) for _, c in cand]
    cols = [set(tokens(c["col"])) for _, c in cand]
    out = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i][1], cand[j][1]
            inter = len(cols[i] & cols[j]) + len(rows[i] & rows[j])
            uni = len(cols[i] | cols[j]) + len(rows[i] | rows[j])
            ov = inter / max(1, uni)
            rat = abs(math.log((abs(a["v"]) + 1e-9) / (abs(b["v"]) + 1e-9)))
            out.append((i, j, [
                u[i], u[j], cs[i], cs[j],
                1.0 if a["r"] == b["r"] else 0.0,
                1.0 if a["c"] == b["c"] else 0.0,
                min(rat, 5.0) / 5.0,
                ov,
                1.0 if (i == 0 or j == 0) else 0.0,
                len(qw & rows[i]) / max(1, len(rows[i])),
                len(qw & rows[j]) / max(1, len(rows[j])),
                len(qw & cols[i]) / max(1, len(cols[i])),
                len(qw & cols[j]) / max(1, len(cols[j])),
            ]))
    return out


def covers(pair, gold):
    return set(gold) <= {(pair[0][1]["r"], pair[0][1]["c"]), (pair[1][1]["r"], pair[1][1]["c"])}


def main():
    ds = build_dataset("../data/dev.json")
    tok = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")

    # 划分：评测集按 key（id + gold）取；训练集取调参段中 key 不在评测集里的样本
    tr, ev, seen_ev = [], [], set()
    for i, d in enumerate(ds):
        try:
            key = (d["id"], round(float(d["exe_ans"]), 6))
        except (TypeError, ValueError):
            continue
        prog = ((RAW.get(d["id"], {}).get("qa") or {}).get("program") or "").strip()
        if not re.match(r"^[a-z]+\s*\([^()]*\)$", prog):
            continue
        if key in EVAL_KEYS:
            if key not in seen_ev:
                seen_ev.add(key)
                ev.append((key, d))
        else:
            tr.append((key, d))
    print("训练 %d / 评测 %d（按 id 切分，无重叠）" % (len(tr), len(ev)), flush=True)

    def prep(items):
        out = []
        for key, d in items:
            r = build(d, tok)
            if not r:
                continue
            doc, cand, gold = r
            qw = set(tokens(d["question"]))
            qvec = S.question(d["question"])
            fs = feats(cand, gold, d, qvec, qw)
            out.append(dict(key=key, cand=cand, gold=gold, fs=fs, q=d["question"]))
        return out

    T, E = prep(tr), prep(ev)
    print("可用：训练 %d、评测 %d" % (len(T), len(E)), flush=True)

    X, y = [], []
    for s in T:
        for i, j, f in s["fs"]:
            X.append(f)
            y.append(1 if covers((s["cand"][i], s["cand"][j]), s["gold"]) else 0)
    X, y = np.array(X), np.array(y)
    print("训练对 %d，正例 %d（%.1f%%）" % (len(X), y.sum(), 100 * y.mean()))

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)

    def sel_ind(s):
        return (s["cand"][0], s["cand"][1])

    def sel_joint(s):
        best, bp = -1e9, None
        for i, j, f in s["fs"]:
            p = clf.predict_proba(np.array([f]))[0, 1]
            if p > best:
                best, bp = p, (s["cand"][i], s["cand"][j])
        return bp

    def sel_joint_rule(s):
        """无训练的联合规则：单元格分 + 互补性 + 问题类型定向。

        增长/变化类问题（"change from 2016 to 2017"）要「同行不同列」；
        占比类问题（"X as a percentage of total"）要「同列不同行」。
        """
        q = s["q"].lower()
        grow = bool(re.search(r"change|increase|decrease|growth|from .* to ", q))
        share = bool(re.search(r"percent of|percentage of|share of|portion|ratio of", q))
        best, bp = -1e9, None
        for i, j, f in s["fs"]:
            a, b = s["cand"][i][1], s["cand"][j][1]
            same_r = 1.0 if a["r"] == b["r"] else 0.0
            same_c = 1.0 if a["c"] == b["c"] else 0.0
            bonus = 2.0 if (same_r + same_c == 1.0) else 0.2
            if grow and same_r and not same_c:
                bonus += 1.5
            if share and same_c and not same_r:
                bonus += 1.5
            sc = f[0] + f[1] + bonus - 0.5 * f[7]
            if sc > best:
                best, bp = sc, (s["cand"][i], s["cand"][j])
        return bp

    def sel_oracle(s):
        gs = set(s["gold"])
        pick = [c for c in s["cand"] if (c[1]["r"], c[1]["c"]) in gs][:2]
        while len(pick) < 2:
            for c in s["cand"]:
                if c not in pick:
                    pick.append(c)
                    break
            else:
                break
        return (pick[0], pick[1]) if len(pick) >= 2 else None

    # 供端到端实验使用：把 Joint-R 选出的对写成 {id|gold: [[r,c],[r,c]]}
    dump = {}
    for s in E:
        pr = sel_joint_rule(s)
        if pr:
            dump["%s|%.6f" % (s["key"][0], s["key"][1])] = [
                [pr[0][1]["r"], pr[0][1]["c"]], [pr[1][1]["r"], pr[1][1]["c"]]]
    json.dump(dump, open("../results/pairs_jointR.json", "w"), ensure_ascii=False)
    dump2 = {}
    for s in E:
        pr = sel_ind(s)
        dump2["%s|%.6f" % (s["key"][0], s["key"][1])] = [
            [pr[0][1]["r"], pr[0][1]["c"]], [pr[1][1]["r"], pr[1][1]["c"]]]
    json.dump(dump2, open("../results/pairs_ind.json", "w"), ensure_ascii=False)
    print("已写出 Joint-R %d 题 / Ind %d 题" % (len(dump), len(dump2)))

    for name, fn in (("Ind", sel_ind), ("Joint-R", sel_joint_rule),
                     ("Joint-L", sel_joint), ("Oracle", sel_oracle)):
        hit = sum(1 for s in E if fn(s) and covers(fn(s), s["gold"]))
        print("%-7s 整对命中 %.3f（%d/%d）" % (name, hit / len(E), hit, len(E)))


if __name__ == "__main__":
    main()
