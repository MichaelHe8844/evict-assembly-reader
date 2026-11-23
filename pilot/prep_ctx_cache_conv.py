# -*- coding: utf-8 -*-
"""预计算 ConvFinQA 训练集在四档预算下的压缩证据（先数值后绑定）。

与 prep_ctx_cache_vf.py 相同，只是数据源换成 ConvFinQA 的展平逐轮文件
（已由 prep_conv.py 拼好对话上下文）。

用法：python prep_ctx_cache_conv.py [limit] [out]
"""
import json
import sys
import time

sys.path.insert(0, ".")
import arm_c                              # noqa: E402
arm_c.VALUE_FIRST = True
from common import build_dataset          # noqa: E402
from lora_budget import keep_text         # noqa: E402
from transformers import AutoTokenizer    # noqa: E402

BUD = ("3", "5", "10", "20")


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    src = "../results/conv_train.json"
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else "../results/ctx_cache_conv.json")
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    rows, seen = [], set()
    for d in build_dataset(src):
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        rows.append(d)
        if lim and len(rows) >= lim:
            break
    print("训练题 %d" % len(rows), flush=True)

    out, t0 = {}, time.time()
    for i, d in enumerate(rows):
        out[d["id"]] = {b: keep_text(tok, d, float(b) / 100.0) for b in BUD}
        if (i + 1) % 500 == 0:
            print("  %d/%d  %.1fs" % (i + 1, len(rows), time.time() - t0), flush=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    print("done %d 题，用时 %.0f 秒" % (len(out), time.time() - t0))


if __name__ == "__main__":
    main()
