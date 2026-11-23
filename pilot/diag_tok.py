# -*- coding: utf-8 -*-
"""分词粒度诊断：同一个「20% 预算」在不同分词器下到底买到多少内容。

动机：预算按 token 比例给，但各族的数字切分粒度差别很大
（Llama-3 三位一组、Qwen 基本逐位）。同一比例下 Llama 保留的数字字符更多，
所以跨族比较必须同时报「有效预算」，否则会把分词器差异误读成模型鲁棒性差异。
"""
import json
import numpy as np
from transformers import AutoTokenizer

from common import build_dataset
from arm_a import tokenize_sample

MODELS = [
    ("Qwen2.5-3B", "E:/KV_models/Qwen2.5-3B-Instruct"),
    ("Llama-3.2-3B", "E:/KV_models/Llama-3.2-3B-Instruct"),
    ("Phi-3.5-mini", "C:/models/Phi-3.5-mini-instruct"),
]

ds = build_dataset("../data/dev.json")[:120]
print("%-14s %8s %10s %10s %10s %10s" % (
    "tokenizer", "doc tok", "tok/char", "cell tok", "cell char", "digit tok"))
out = {}
for name, path in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print("%-14s 加载失败 %s" % (name, str(e)[:40]))
        continue
    Ls, cell_tok, cell_chr, dig_tok, dig_chr = [], [], [], [], []
    for d in ds:
        ids, spans = tokenize_sample(tok, d["segs"])
        Ls.append(len(ids))
        for s, (a, b) in zip(d["segs"], spans):
            if s["kind"] != "cell" or a < 0:
                continue
            cell_tok.append(b - a)
            cell_chr.append(len(s["text"]))
            # 数字本身的切分粒度：只数数字字符对应的 token 数
            n_dig = sum(ch.isdigit() for ch in s["text"])
            if n_dig:
                dig_tok.append(b - a)
                dig_chr.append(n_dig)
    print("%-14s %8.0f %10.3f %10.2f %10.1f %10.2f" % (
        name, np.mean(Ls), np.mean(Ls) / np.mean([sum(len(s["text"]) for s in d["segs"]) for d in ds]),
        np.mean(cell_tok), np.mean(cell_chr),
        sum(dig_tok) / max(1, sum(dig_chr)) * 1.0))
    out[name] = dict(doc_tok=float(np.mean(Ls)), cell_tok=float(np.mean(cell_tok)),
                     cell_char=float(np.mean(cell_chr)),
                     cell_tok_per_char=float(np.mean(np.array(cell_tok) / np.array(cell_chr))))

json.dump(out, open("../results/tok_granularity.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n写入 ../results/tok_granularity.json")
