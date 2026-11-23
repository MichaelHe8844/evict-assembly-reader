# -*- coding: utf-8 -*-
"""统一的模型加载入口：支持 bf16 与 4-bit（bitsandbytes）。

Arm B 需要 output_attentions=True，所以必须用 eager attention；
4-bit 量化只压权重，注意力权重仍在计算 dtype 上，因此两者可共存（已验证）。
"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_model(path, quant4=False, device="cuda", quant8=False, attn="eager"):
    kw = dict(attn_implementation=attn, device_map=device)
    if quant4:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, **kw)
    if quant8:
        # 8-bit 是长上下文的关键：权重 ~7.6GB，L=1400 峰值 9.3GB、耗时 0.9s；
        # 而 bf16 在 L=1400 会触发 WDDM 换页，慢 200 倍以上。
        return AutoModelForCausalLM.from_pretrained(
            path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), **kw)
    for dt in ("dtype", "torch_dtype"):
        try:
            return AutoModelForCausalLM.from_pretrained(path, **{dt: torch.bfloat16}, **kw)
        except TypeError:
            continue
    return AutoModelForCausalLM.from_pretrained(path, **kw)
