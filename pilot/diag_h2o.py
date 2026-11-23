# -*- coding: utf-8 -*-
"""
H2O 实现自检：打开一个样本，看它到底保留了哪些 token。

疑点：H2O@20 的 NumR 只有 0.013（中位 0），比近期窗口（0.20）低一个数量级。
在把 0.033 的 EM 写进论文前，必须先确认这不是打分/索引写错。
"""
import json
import numpy as np
import torch

from common import build_dataset
from arm_a import tokenize_sample
from model_utils import load_model
from pot_prompt import FEWSHOT, INSTR
from transformers import AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    model = load_model("E:/KV_models/Qwen2.5-3B-Instruct")
    model.eval()

    ds = build_dataset("../data/dev.json")
    d = ds[0]
    ids_doc, spans = tokenize_sample(tok, d["segs"])
    n_doc = len(ids_doc)

    # 与 arm_c 一致的构造：前言 + 文档 + 后缀（问题）
    pre = tok(FEWSHOT, add_special_tokens=False)["input_ids"]
    suf = tok(INSTR + "\n" + d["question"], add_special_tokens=False, return_tensors="pt")
    suf_ids = tok(INSTR + "\n" + d["question"], add_special_tokens=False)["input_ids"]
    ids = pre + ids_doc + suf_ids
    off = len(pre)
    L = len(ids)

    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device="cuda"),
                  output_attentions=True, use_cache=True)
    atts = o.attentions
    snap = torch.zeros(n_doc, device="cuda")
    h2o = torch.zeros(n_doc, device="cuda")
    W = min(64, len(suf_ids))
    for a in atts:
        a = a[0].float()
        snap += a[:, -W:, off:off + n_doc].sum(dim=(0, 1))
        h2o += a[:, :, off:off + n_doc].sum(dim=(0, 1))
    snap = snap.cpu().numpy()
    h2o = h2o.cpu().numpy()

    B = max(8, int(0.20 * n_doc))
    print("L=%d  n_doc=%d  B=%d" % (L, n_doc, B))

    # 每个 token 落在哪个片段
    owner = np.full(n_doc, -1)
    for si, (a, b) in enumerate(spans):
        if a >= 0:
            owner[a:b] = si

    for name, score in (("H2O", h2o), ("SnapKV", snap)):
        keep = np.zeros(n_doc, dtype=bool)
        keep[:4] = True
        keep[np.argsort(-score)[:B]] = True
        kept_segs = [si for si in set(owner[keep].tolist()) if si >= 0]
        kinds = {}
        for si in kept_segs:
            k = d["segs"][si]["kind"]
            kinds[k] = kinds.get(k, 0) + 1
        # 数值格是否保留
        cell_kept = sum(1 for si in kept_segs if d["segs"][si]["kind"] == "cell")
        cell_all = sum(1 for s in d["segs"] if s["kind"] == "cell")
        print("\n== %s：保留 %d 片段，其中 cell %d / 全表 %d" % (name, len(kept_segs), cell_kept, cell_all))
        print("   片段类型分布:", kinds)
        txt = " ".join(d["segs"][si]["text"] for si in sorted(kept_segs)[:18])
        print("   前 18 个保留片段:", txt[:220])


if __name__ == "__main__":
    main()
