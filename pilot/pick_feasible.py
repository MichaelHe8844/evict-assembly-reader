# -*- coding: utf-8 -*-
"""学习组件：逐格打分 + 预算可行的子集选择（BFOPS）。

与旧流程的区别：
  旧：成对模型给 28 对打分 → argmax（不管装不装得下）→ 装不下按三级规则回退
  新：单格交叉编码器给 8 个候选格打分 → 在「并集代价 ≤ B」的可行子集里取总分最高者
      目标  max_{|S|<=2} sum_{i in S} s_i   s.t. |∪ T_i| <= B
      枚举 36 个子集（28 对 + 8 单格），精确最优；装不下整对时自动退到能装下的最好单格

输出：与 arm_c --force_json 兼容的 { "id|gold": [[r,c], ...] }，
      以及逐个样本的诊断（选了 1 格还是 2 格、代价、是否可行）。

用法：
  python pick_feasible.py --data ../data/dev.json --pairs ../results/r1_pairs_dev_full.json \
      --fracs 3,5,10,20 --tag af
"""
import argparse
import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, ".")
from arm_a import tokenize_sample          # noqa: E402
from common import build_dataset           # noqa: E402
from r1_cell import M, make_ids            # noqa: E402

BASE = "E:/KV_models/bge-small-en-v1.5"


def load_scorer(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hidden = ck["state"]["head.0.weight"].shape[1]
    nf = hidden - AutoModel.from_pretrained(ck["base"]).config.hidden_size
    m = M(ck["base"], nf=nf).cuda()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, nf


def cell_tokens(segs, spans, r, c):
    """一个候选格的锚点组 token 下标：数值 + 行头 + 列名 + 单位。"""
    out = set()
    for si, s in enumerate(segs):
        a, b = spans[si]
        if a < 0 or b <= a:
            continue
        k = s.get("kind")
        if k == "cell" and s.get("row") == r and s.get("col") == c:
            out |= set(range(a, b))
        elif k == "row_header" and s.get("row") == r:
            out |= set(range(a, b))
        elif k in ("col_name", "col_header", "unit") and s.get("col") == c:
            out |= set(range(a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/dev.json")
    ap.add_argument("--pairs", default="../results/r1_pairs_dev_full.json")
    ap.add_argument("--scorer", default="../results/ce_cell.pt")
    ap.add_argument("--fracs", default="3,5,10,20")
    ap.add_argument("--tag", default="af")
    ap.add_argument("--maxlen", type=int, default=1500)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    model, nf = load_scorer(args.scorer)
    rows = json.load(open(args.pairs, encoding="utf-8"))
    ds = {}
    for d in build_dataset(args.data):
        ds[d["id"]] = d
    print("候选 %d 题，读到 %d 个文档" % (len(rows), len(ds)), flush=True)

    fracs = [float(x) / 100 for x in args.fracs.split(",")]
    force = {f: {} for f in fracs}
    diag = {f: {} for f in fracs}
    miss = 0
    for n, r in enumerate(rows):
        qid = r["qid"]
        d = ds.get(qid)
        if d is None:
            miss += 1
            continue
        gold = float(r["key"][1])
        doc, sp = tokenize_sample(tok, d["segs"])
        if len(doc) > args.maxlen:
            continue
        # 逐格打分（一次前向一格）
        scores = []
        with torch.no_grad():
            for c in r["cands"]:
                ids = torch.tensor([make_ids(tok, r["q"], [c["txt"]])]).cuda()
                F = torch.zeros(1, nf, device="cuda")
                scores.append(float(model(ids, torch.ones_like(ids), F)[0]))
        toks = [cell_tokens(d["segs"], sp, c["r"], c["c"]) for c in r["cands"]]
        for f in fracs:
            B = max(8, int(f * len(doc)))
            # 字典序：只要存在可行的一对，就必须选一对（任务的语义就是两个操作数）；
            # 只有在「没有任何一对装得下」时才退到单格。对内的选择取分数和最大。
            pick, cost, bsc = None, 0, -1e9
            for i in range(len(r["cands"])):
                for j in range(i + 1, len(r["cands"])):
                    u = toks[i] | toks[j]
                    if len(u) <= B and scores[i] + scores[j] > bsc:
                        pick, cost, bsc = [i, j], len(u), scores[i] + scores[j]
            if pick is None:
                for i in range(len(r["cands"])):
                    if len(toks[i]) <= B and scores[i] > bsc:
                        pick, cost, bsc = [i], len(toks[i]), scores[i]
            if pick is None:          # 连单格都装不下：交给 arm_c 的回退链处理
                pick = [max(range(len(scores)), key=lambda k: scores[k])]
                cost = len(toks[pick[0]])
            key = "%s|%.6f" % (qid, round(gold, 6))
            force[f][key] = [[r["cands"][i]["r"], r["cands"][i]["c"]] for i in pick]
            diag[f][key] = dict(n=len(pick), cost=int(cost), B=int(B),
                                feasible=int(cost <= B),
                                nfeas=int(sum(1 for i in range(len(toks))
                                              if len(toks[i]) <= B)))
        if (n + 1) % 50 == 0:
            print("  %d/%d" % (n + 1, len(rows)), flush=True)
    print("缺文档 %d 条" % miss, flush=True)

    for f in fracs:
        p = "../results/fp_%s%d.json" % (args.tag, round(f * 100))
        json.dump(force[f], open(p, "w", encoding="utf-8"), ensure_ascii=False)
        dd = diag[f]
        two = sum(1 for v in dd.values() if v["n"] == 2)
        feas = sum(1 for v in dd.values() if v["feasible"] and v["n"] == 2)
        print("%s：%d 题，选 2 格 %d（%.1f%%），其中完整装入 %d（%.1f%%），"
              "平均代价 %.1f / 预算 %.1f"
              % (p, len(dd), two, 100 * two / max(1, len(dd)),
                 feas, 100 * feas / max(1, len(dd)),
                 sum(v["cost"] for v in dd.values()) / max(1, len(dd)),
                 sum(v["B"] for v in dd.values()) / max(1, len(dd))), flush=True)
        json.dump(dd, open(p.replace(".json", "_diag.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
