# -*- coding: utf-8 -*-
"""
长上下文可行性台架：16G 卡上，7B 8-bit 能吃到多长？

要回答两件事：
  1. output_attentions=True（现在 arm_c 的做法）在长上下文下是否必然爆显存
     —— 它会把每层 L×L 的注意力矩阵都留下来。
  2. 换成 sdpa 且不取注意力，L=8k/12k/16k 的 prefill 时间与峰值显存是多少。

用法：python bench_long.py --model E:/KV_models/Qwen2.5-7B-Instruct --lens 8000 12000
"""
import argparse
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def tok_len(tok, n_target):
    """造一段约 n_target token 的重复文本。"""
    unit = ("the company reported net sales of 1234 million in fiscal 2016 , "
            "compared with 1102 million in fiscal 2015 , an increase of 12.0 percent . ")
    reps = max(1, n_target // 25)
    return tok(unit * reps, add_special_tokens=False, return_tensors="pt")["input_ids"]


def run(model, ids, output_attn):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        o = model(input_ids=ids, output_attentions=output_attn, use_cache=True,
                  logits_to_keep=1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del o
    torch.cuda.empty_cache()
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lens", type=int, nargs="+", default=[4000, 8000, 12000])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--load_8bit", action="store_true", default=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    from model_utils import load_model
    model = load_model(args.model, quant8=True)
    if args.attn != "eager":
        model.config._attn_implementation = args.attn
    model.eval()
    print("attn_implementation =", args.attn, "| 参数显存 %.2f GB"
          % (torch.cuda.memory_allocated() / 1024 ** 3))

    print("\n%-8s %-14s %10s %10s %s" % ("L", "output_attentions", "用时(s)", "峰值(GB)", "结果"))
    for L in args.lens:
        ids = tok_len(tok, L).to("cuda")
        real = ids.shape[1]
        for oa in (False, True):
            try:
                dt, peak = run(model, ids, oa)
                print("%-8d %-14s %10.1f %10.2f %s" % (real, str(oa), dt, peak, "ok"))
            except torch.cuda.OutOfMemoryError:
                print("%-8d %-14s %10s %10s %s" % (real, str(oa), "-", "-", "OOM"))
                torch.cuda.empty_cache()
            except Exception as e:
                print("%-8d %-14s %10s %10s %s" % (real, str(oa), "-", "-", type(e).__name__))
                torch.cuda.empty_cache()
        del ids


if __name__ == "__main__":
    main()
