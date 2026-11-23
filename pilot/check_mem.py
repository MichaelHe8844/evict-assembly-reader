# -*- coding: utf-8 -*-
"""
验证：7B bf16 在 1400 token 下，
  (a) 常规 output_attentions=True 是否 OOM；
  (b) 逐层归约（拦截 eager_attention_forward，边算边压成向量）能否跑下。

(b) 的原理：output_attentions=True 会把 28 层的注意力矩阵**同时**留在显存里
(28×28×L²)，而它们只需要各自归约成 (L,) 的重要性向量。逐层归约的峰值是
**单层**矩阵，L=1400 时从 3.1GB 降到约 110MB。
"""
import sys
import torch
import transformers.models.qwen2.modeling_qwen2 as qm
from transformers import AutoModelForCausalLM, AutoTokenizer

P = "E:/KV_models/Qwen2.5-7B-Instruct"
MODE = sys.argv[1] if len(sys.argv) > 1 else "patch"
L_TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 1400

tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map="cuda",
                                         attn_implementation="eager").eval()
print("weights %.2f GB / total %.2f GB" % (
    torch.cuda.memory_allocated() / 1e9,
    torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)

unit = "the company reported net revenue of 2450 million in fiscal 2022 ; "
ids = tok(unit, add_special_tokens=False)["input_ids"]
ids = (ids * (L_TARGET // len(ids) + 1))[:L_TARGET]
x = torch.tensor([ids], device="cuda")
L = x.shape[1]
print("L =", L, flush=True)

ACC = {"v": None, "n": 0}


def patch():
    orig = qm.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out = orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        attn_output, attn_weights = out[0], out[1]
        if attn_weights is not None:
            w = attn_weights[0]                      # (H,Q,K) bf16
            red = w[:, -64:, :].sum(dim=(0, 1), dtype=torch.float32)
            ACC["v"] = red if ACC["v"] is None else ACC["v"] + red
            ACC["n"] += 1
            del w, red
            attn_weights = None                      # 不让上层累积
        return (attn_output, attn_weights)

    qm.eager_attention_forward = patched


if MODE == "patch":
    patch()

torch.cuda.reset_peak_memory_stats()
try:
    with torch.no_grad():
        o = m(input_ids=x, output_attentions=(MODE == "plain"), use_cache=False)
    print("OK  peak %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    if MODE == "patch":
        print("归约层数 %d, 向量长度 %d, 和为 %.3f" % (
            ACC["n"], len(ACC["v"]), float(ACC["v"].sum())), flush=True)
except torch.cuda.OutOfMemoryError as e:
    print("OOM:", str(e)[:200], flush=True)
