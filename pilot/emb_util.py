# -*- coding: utf-8 -*-
"""把 bge-small 的句向量相似度封装成抽取器可用的打分函数。

只做前向、不训练：表头向量按文本缓存（同一份文档里表头重复率高），
问题向量每条算一次。单条样本的额外开销约 3–6 ms。
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

INSTR = "Represent this sentence for searching relevant passages: "


class EmbScorer:
    def __init__(self, path="E:/KV_models/bge-small-en-v1.5", device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.dev = device
        self.cache = {}

    @torch.no_grad()
    def _enc(self, texts, is_query=False, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            chunk = [(INSTR + t) if is_query else t for t in texts[i:i + batch]]
            b = self.tok(chunk, padding=True, truncation=True, max_length=64,
                         return_tensors="pt").to(self.dev)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float32))
        return np.concatenate(out, 0)

    def question(self, q):
        return self._enc([q], is_query=True)[0]

    def header(self, h):
        v = self.cache.get(h)
        if v is None:
            v = self._enc([h])[0]
            self.cache[h] = v
        return v

    def scorer(self, q):
        qv = self.question(q)

        def f(h):
            return float(max(0.0, float(qv @ self.header(h))))

        return f
