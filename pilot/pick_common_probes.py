# -*- coding: utf-8 -*-
"""挑一批「三个分词器下都不超过 maxlen」的公共探针。

不这么做的话，每个模型按自己的分词器做长度过滤，截出来的 150 条不是同一批
（实测 Qwen 80 / Llama 78 / Phi 82 个不同 id），跨族比较就混入了样本差异。
"""
import json
from transformers import AutoTokenizer

from common import build_dataset, norm_num
from arm_a import tokenize_sample

MAXLEN = 1200
WANT = 150
MODELS = {
    "qwen": "E:/KV_models/Qwen2.5-3B-Instruct",
    "llama": "E:/KV_models/Llama-3.2-3B-Instruct",
    "phi": "C:/models/Phi-3.5-mini-instruct",
}

toks = {k: AutoTokenizer.from_pretrained(v) for k, v in MODELS.items()}
ds = build_dataset("../data/dev.json")

picked, seen = [], set()
for d in ds:
    if len(picked) >= WANT:
        break
    if not (d["group"].get("C") and d["group"].get("R")):
        continue
    if norm_num(d["gold_value"]) is None:
        continue
    key = [d["id"], d["row"], d["col"]]
    if tuple(key) in seen:
        continue
    lens = {}
    for k, tok in toks.items():
        ids, _ = tokenize_sample(tok, d["segs"])
        lens[k] = len(ids)
    if max(lens.values()) > MAXLEN:
        continue
    seen.add(tuple(key))
    picked.append(dict(key=key, lens=lens))

json.dump(picked, open("../results/xmodel_keys.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
n_doc = len({p["key"][0].rsplit("-", 1)[0] for p in picked})
print("挑出 %d 条，覆盖 %d 个文档" % (len(picked), n_doc))
for k in MODELS:
    v = [p["lens"][k] for p in picked]
    print("  %-6s 长度 中位 %d 最大 %d" % (k, sorted(v)[len(v) // 2], max(v)))
