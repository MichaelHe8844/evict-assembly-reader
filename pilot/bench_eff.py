# -*- coding: utf-8 -*-
"""
效率实测：20% 的 token 保留率到底换来多少显存与速度？

对同一批样本测三件事：
  1. KV 缓存字节数（全上下文 / 裁到 20%）
  2. 峰值显存（prefill 阶段，含 8-bit 权重）
  3. prefill 时延与解码速度（tok/s）
另外单独测锚点抽取（选点）的 CPU 开销。

用法：python bench_eff.py --limit 10
"""
import argparse
import copy
import json
import re
import time

import numpy as np
import torch

from common import build_dataset
from anchor_auto import extract
from arm_a import tokenize_sample
from arm_c import _nums, fit_budget
from model_utils import load_model
from transformers import AutoTokenizer

SINGLE = re.compile(r"^[a-z]+\s*\([^()]*\)$")
PRIO = {"col_name": 0, "unit": 0, "col_header": 0, "row_header": 1, "cell": 2, "text": 3, "sep": 9}
SYS = "You are a financial analyst."
INSTR = "\nAnswer with the final number only."


def _tensors(pkv):
    """兼容各版本 cache：layers / key_cache / tuple。"""
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            if getattr(layer, "keys", None) is not None:
                yield layer.keys
                yield layer.values
        return
    if hasattr(pkv, "key_cache"):
        for k, v in zip(pkv.key_cache, pkv.value_cache):
            yield from (k, v)
        return
    for layer in pkv:
        if layer is None:
            continue
        if torch.is_tensor(layer):
            yield layer
            continue
        for t in layer:
            if t is not None:
                yield t


def kv_bytes(pkv):
    return sum(t.numel() * t.element_size() for t in _tensors(pkv))


def trim(pkv, keep):  # keep: np.array of positions (sorted)
    if hasattr(pkv, "layers"):
        idx = torch.as_tensor(keep, dtype=torch.long, device="cuda")
        for layer in pkv.layers:
            if getattr(layer, "keys", None) is None:
                continue
            layer.keys = layer.keys.index_select(2, idx).contiguous()
            layer.values = layer.values.index_select(2, idx).contiguous()
        return pkv
    if hasattr(pkv, "key_cache"):
        from transformers.cache_utils import DynamicCache
        c = DynamicCache()
        for k, v in zip(pkv.key_cache, pkv.value_cache):
            idx = torch.as_tensor(keep, dtype=torch.long, device=k.device)
            c.update(k.index_select(2, idx), v.index_select(2, idx), len(c))
        return c
    first = next(_tensors(pkv))
    idx = torch.as_tensor(keep, dtype=torch.long, device=first.device)
    return tuple(tuple(t.index_select(2, idx) if t is not None else None for t in layer)
                 for layer in pkv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E:/KV_models/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--n_new", type=int, default=16)
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--maxlen", type=int, default=1100)
    ap.add_argument("--out", default="../results/bench_eff.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, False, quant8=True).eval()
    ds = build_dataset("../data/dev.json")
    raw = {e["id"]: e for e in json.load(open("../data/dev.json", encoding="utf-8"))}

    rows = []
    seen = set()
    for d in ds[400:]:
        if len(rows) >= args.limit:
            break
        qa = (raw.get(d["id"], {}) or {}).get("qa") or {}
        if not SINGLE.match((qa.get("program") or "").strip()):
            continue
        if (d["id"], d["row"], d["col"]) in seen:
            continue
        seen.add((d["id"], d["row"], d["col"]))
        ids, sp = tokenize_sample(tok, d["segs"])
        if not (64 <= len(ids) <= args.maxlen):
            continue
        rows.append((d, ids, sp))

    print("样本 %d" % len(rows))
    recs = []
    for k, (d, doc_ids, sp) in enumerate(rows, 1):
        pre = tok("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n" % SYS,
                  add_special_tokens=False)["input_ids"]
        suf = tok("\n\nQuestion: %s\n%s<|im_end|>\n<|im_start|>assistant\n"
                  % (d["question"], INSTR), add_special_tokens=False)["input_ids"]
        ids = pre + doc_ids + suf
        off, n_doc = len(pre), len(doc_ids)
        L = len(ids)

        # ---- 选点开销（CPU）----
        t0 = time.time()
        idx, _ = extract(d["segs"], d["question"])
        extract_ms = (time.time() - t0) * 1000

        # ---- prefill（全上下文）----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        prefill_s = time.time() - t0
        peak_full = torch.cuda.max_memory_allocated() / 1e9
        pkv = o.past_key_values
        kv_full = kv_bytes(pkv)
        del o

        # ---- 本文 20%：锚点 + 填充 ----
        B = max(8, int(args.budget * n_doc))
        keep = set(fit_budget(sorted([i for i in idx if sp[i][0] >= 0],
                                     key=lambda i: PRIO.get(d["segs"][i]["kind"], 9)),
                              d["segs"], [(a + off, b + off) if a >= 0 else (-1, -1) for a, b in sp], B))
        # 前缀与后缀永不裁
        keep_all = np.array(sorted(set(range(off)) | {t + off for t in keep}
                                   | set(range(off + n_doc, L))), dtype=np.int64)
        pkv_trim = trim(copy.deepcopy(pkv), keep_all)
        kv_trim = kv_bytes(pkv_trim)

        def decode(pkv_, n):
            cur = torch.tensor([[ids[-1]]], device="cuda")
            torch.cuda.synchronize(); t = time.time()
            with torch.no_grad():
                for _ in range(n):
                    out = model(input_ids=cur, past_key_values=pkv_, use_cache=True)
                    pkv_ = out.past_key_values
                    cur = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            return n / (time.time() - t)

        tps_full = decode(copy.deepcopy(pkv), args.n_new)
        tps_trim = decode(pkv_trim, args.n_new)
        del pkv
        torch.cuda.empty_cache()

        recs.append(dict(id=d["id"], L=L, n_doc=n_doc, extract_ms=extract_ms,
                         prefill_s=prefill_s, peak_full=peak_full,
                         kv_full=kv_full, kv_trim=kv_trim,
                         tps_full=tps_full, tps_trim=tps_trim))
        print("  [%d/%d] L=%d 选点 %.1fms | prefill %.2fs | KV %.0f→%.0f MB (%.1f%%) | "
              "峰值 %.2f GB | 解码 %.1f→%.1f tok/s"
              % (k, len(rows), L, extract_ms, prefill_s, kv_full / 1e6, kv_trim / 1e6,
                 100 * kv_trim / kv_full, peak_full, tps_full, tps_trim))

    def agg(k):
        v = np.array([r[k] for r in recs], dtype=float)
        return float(np.median(v)), float(np.mean(v))

    print("\n=== 汇总（中位 / 均值）===")
    for k, name, unit in (("extract_ms", "选点开销", "ms"), ("prefill_s", "prefill", "s"),
                          ("peak_full", "峰值显存", "GB"), ("kv_full", "KV 全量", "MB"),
                          ("kv_trim", "KV 裁后", "MB"), ("tps_full", "解码(全量)", "tok/s"),
                          ("tps_trim", "解码(裁剪)", "tok/s")):
        m, a = agg(k)
        print("  %-12s %.2f / %.2f %s" % (name, m / (1e6 if unit == "MB" else 1), a / (1e6 if unit == "MB" else 1), unit))
    kvf, pkv_ = agg("kv_full")[1], agg("peak_full")[1]
    print("  KV 保留比例 %.1f%% | 全量 KV 占峰值显存 %.2f%% ⇒ 理论最多省 %.2f GB"
          % (100 * agg("kv_trim")[1] / kvf, 100 * kvf / (pkv_ * 1e9), kvf * (1 - agg("kv_trim")[1] / kvf) / 1e9))

    json.dump(dict(records=recs, n=len(recs)), open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("写入", args.out)


if __name__ == "__main__":
    main()
