# -*- coding: utf-8 -*-
"""跨数据集准备：把 FinQA test 划分（字段与 dev 完全一致）整理成
固定题单 + 成对数据，供选择器零样本迁移与端到端实验复用。

FinQA test 与 dev/train **文档零重叠**（380 份文档都没用过）。
"""
import json
import re
import sys

sys.path.insert(0, ".")
from common import build_dataset  # noqa: E402
from arm_a import tokenize_sample  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from r1_data import build_pairs  # noqa: E402

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")
SRC = "../data/fin_test.json"


def main():
    maxlen = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
    raw = {e["id"]: e for e in json.load(open(SRC, encoding="utf-8"))}
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")

    seen, keys = set(), []
    for d in build_dataset(SRC):
        e = raw.get(d["id"])
        if not e:
            continue
        prog = ((e.get("qa") or {}).get("program") or "").strip()
        if not SINGLE.match(prog):
            continue
        try:
            k = (d["id"], round(float(d["exe_ans"]), 6))
        except (TypeError, ValueError):
            continue
        if k in seen:
            continue
        seen.add(k)
        doc, _ = tokenize_sample(tok, d["segs"])
        if 64 <= len(doc) <= maxlen:
            keys.append(k)

    print("test 单步可用（<=%d token）：%d 题 / %d 文档"
          % (maxlen, len(keys), len({k[0].rsplit('-', 1)[0] for k in keys})))
    json.dump([{"key": [a, b]} for a, b in keys],
              open("../results/xds_keys.json", "w"), ensure_ascii=False)

    rows = build_pairs(SRC, raw, only_keys=set(keys), budget=None, tok=tok)
    json.dump(rows, open("../results/xds_pairs_test.json", "w"), ensure_ascii=False)
    npos = sum(p[2] for r in rows for p in r["pairs"])
    print("成对数据：%d 题 / %d 对 / 正例 %d（池内可解率 %.3f）"
          % (len(rows), sum(len(r["pairs"]) for r in rows), npos,
             sum(any(p[2] for p in r["pairs"]) for r in rows) / max(1, len(rows))))


if __name__ == "__main__":
    main()
