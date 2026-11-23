# -*- coding: utf-8 -*-
"""计时：逐层归约下，7B bf16 在不同上下文长度上的前向耗时。"""
import time
import torch
import transformers.models.qwen2.modeling_qwen2 as qm
from transformers import AutoModelForCausalLM, AutoTokenizer

P = "E:/KV_models/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map="cuda",
                                         attn_implementation="eager").eval()

ACC = {"v": None, "n": 0}
orig = qm.eager_attention_forward


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    out = orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
    a_out, a_w = out[0], out[1]
    if a_w is not None:
        w = a_w[0]
        red = w[:, -64:, :].sum(dim=(0, 1), dtype=torch.float32)
        ACC["v"] = red if ACC["v"] is None else ACC["v"] + red
        ACC["n"] += 1
        del w, red
        a_w = None
    return (a_out, a_w)


qm.eager_attention_forward = patched

unit = "the company reported net revenue of 2450 million in fiscal 2022 ; "
base = tok(unit, add_special_tokens=False)["input_ids"]

for L in (500, 700, 1000, 1400):
    ACC["v"] = None
    ACC["n"] = 0
    ids = (base * (L // len(base) + 1))[:L]
    x = torch.tensor([ids], device="cuda")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        m(input_ids=x, output_attentions=False, use_cache=False)
    torch.cuda.synchronize()
    t1 = time.time()
    # 第二次（缓存/分配器已热）
    with torch.no_grad():
        m(input_ids=x, output_attentions=False, use_cache=False)
    torch.cuda.synchronize()
    t2 = time.time()
    print("L=%4d  首次 %6.1fs  第二次 %6.1fs  峰值 %5.2f GB" % (
        L, t1 - t0, t2 - t1, torch.cuda.max_memory_allocated() / 1e9), flush=True)
