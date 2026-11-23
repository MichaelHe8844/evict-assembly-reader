# -*- coding: utf-8 -*-
"""验证 KV cache 手术的正确性：keep=全部时，手写贪心解码必须与原生 generate 完全一致。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = "E:/KV_models/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map="cuda",
                                         attn_implementation="eager").eval()


def trim(cache, idx):
    idx = torch.as_tensor(idx, device="cuda", dtype=torch.long)
    for layer in cache.layers:
        if layer.keys is None:
            continue
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
    return cache


def greedy(model, cache, logits, start_pos, n_new=16):
    out_toks, pos = [], start_pos
    for _ in range(n_new):
        nxt = int(logits[0, -1].argmax())
        out_toks.append(nxt)
        if nxt == tok.eos_token_id:
            break
        o = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                  past_key_values=cache,
                  position_ids=torch.tensor([[pos]], device="cuda"),
                  use_cache=True)
        cache, logits = o.past_key_values, o.logits
        pos += 1
    return out_toks


prompt = ("Read the financial document and answer.\n\n"
          "company | revenue\nacme | 1234\nbeta | 987\n\n"
          "Question: what is the revenue of acme?\nAnswer:")
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
L = ids.shape[1]

# 原生 generate
with torch.no_grad():
    ref = m.generate(ids, max_new_tokens=16, do_sample=False,
                     pad_token_id=tok.eos_token_id)
ref_new = ref[0, L:].tolist()

# 手术路径：prefill → trim(全部) → 手写贪心
with torch.no_grad():
    o = m(input_ids=ids, use_cache=True)
cache = trim(o.past_key_values, list(range(L)))
with torch.no_grad():
    new = greedy(m, cache, o.logits, L)

print("ref :", ref_new)
print("new :", new[:len(ref_new)])
print("一致:", new[:len(ref_new)] == ref_new)
print("解码文本:", tok.decode(new))

# 再验证：裁掉一半后仍能跑通、位置编码不报错
with torch.no_grad():
    o2 = m(input_ids=ids, use_cache=True)
keep = list(range(0, 4)) + list(range(L // 2, L))
c2 = trim(o2.past_key_values, keep)
with torch.no_grad():
    new2 = greedy(m, c2, o2.logits, L)
print("半裁后:", tok.decode(new2), "| cache 长度:", c2.get_seq_length())
