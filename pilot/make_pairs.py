# -*- coding: utf-8 -*-
"""为扩样本后的评测集批量生成两种候选对（Ind / Joint-R）与 keys 文件。

不训练：Joint-R 是纯规则，扩样本不需要重训。Joint-L 因训练集重叠，不在扩样本上报告。
"""
import json
import re
import sys

sys.path.insert(0, ".")
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from anchor_auto import extract  # noqa: E402
from arm_c import fit_budget  # noqa: E402
from ground import cells_from_segs, rank, tokens  # noqa: E402
from operand_loc import gold_cells  # noqa: E402

MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
RAW = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}
SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")

import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")


def joint_rule(cand, q, qvec=None):
    s = cand
    ql = q.lower()
    grow = bool(re.search(r"change|increase|decrease|growth|from .* to ", ql))
    share = bool(re.search(r"percent of|percentage of|share of|portion|ratio of", ql))
    u = np.array([x[0] for x in s], dtype=np.float32)
    u = u / max(1e-6, u.max())
    best, bp = -1e9, None
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = s[i][1], s[j][1]
            sr = 1.0 if a["r"] == b["r"] else 0.0
            sc = 1.0 if a["c"] == b["c"] else 0.0
            bo = 2.0 if (sr + sc == 1.0) else 0.2
            if grow and sr and not sc:
                bo += 1.5
            if share and sc and not sr:
                bo += 1.5
            ta, tb = set(tokens(a["row"])) | set(tokens(a["col"])), set(tokens(b["row"])) | set(tokens(b["col"]))
            ov = len(ta & tb) / max(1, len(ta | tb))
            v = u[i] + u[j] + bo - 0.5 * ov
            if v > best:
                best, bp = v, (a, b)
    return bp


keys, pj, pi = [], {}, {}
ds = build_dataset("../data/dev.json")
seen = set()
for d in ds:
    e = RAW.get(d["id"])
    if not e or not SINGLE.match(((e.get("qa") or {}).get("program") or "").strip()):
        continue
    try:
        gold = float(d["exe_ans"])
    except (TypeError, ValueError):
        continue
    k = (d["id"], round(gold, 6))
    if k in seen:
        continue
    doc, spans = tokenize_sample(tok, d["segs"])
    if len(doc) < 64 or len(doc) > MAXLEN:
        continue
    mc = max(1, int((0.85 * int(0.20 * len(doc))) / 6))
    idx, _ = extract(d["segs"], d["question"], max_cells=mc)
    B = max(8, int(0.20 * len(doc)))
    keep = set(fit_budget([i for i in idx if spans[i][0] >= 0], d["segs"], spans, B))
    for cross in (False, True):
        pass
    cand = rank(cells_from_segs(d["segs"], spans, keep), d["question"], topk=8)
    if len(cand) < 2:
        continue
    seen.add(k)
    kk = "%s|%.6f" % (k[0], k[1])
    keys.append([k[0], k[1]])
    pi[kk] = [[cand[0][1]["r"], cand[0][1]["c"]], [cand[1][1]["r"], cand[1][1]["c"]]]
    bp = joint_rule(cand, d["question"])
    if bp:
        pj[kk] = [[bp[0]["r"], bp[0]["c"]], [bp[1]["r"], bp[1]["c"]]]

json.dump(keys, open("../results/wide_keys.json", "w"), ensure_ascii=False)
json.dump(pi, open("../results/pairs_wide_ind.json", "w"), ensure_ascii=False)
json.dump(pj, open("../results/pairs_wide_joint.json", "w"), ensure_ascii=False)
print("keys %d | Ind %d | Joint-R %d（maxlen=%d）" % (len(keys), len(pi), len(pj), MAXLEN))
