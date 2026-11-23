# -*- coding: utf-8 -*-
"""预计算：训练集每道单步题在四档预算下的压缩证据文本（先数值后绑定）。

与 prep_ctx_cache.py 的唯一差别：装配优先级改成 4.2 的链
（cell -> row_header -> col_name/col_header/unit），即 arm_c.VALUE_FIRST = True。
输出 results/ctx_cache_train_vf.json，格式与原缓存一致。

用法：
  python prep_ctx_cache_vf.py [limit]
"""
import json
import sys
import time

sys.path.insert(0, ".")
import arm_c                              # noqa: E402
arm_c.VALUE_FIRST = True                  # 先数值、后绑定
from common import build_dataset          # noqa: E402
from lora_budget import keep_text         # noqa: E402
from transformers import AutoTokenizer    # noqa: E402

BUD = ("3", "5", "10", "20")


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else "../results/ctx_cache_train_vf.json")
    tok = AutoTokenizer.from_pretrained("E:/KV_models/Qwen2.5-3B-Instruct")
    raw = json.load(open("../data/train.json", encoding="utf-8"))
    single = set()
    for x in raw:
        qa = x.get("qa") or {}
        if len(qa.get("steps") or []) == 1 and (qa.get("program") or "").strip():
            single.add(x["id"])
    rows, seen = [], set()
    for d in build_dataset("../data/train.json"):
        if d["id"] not in single or d["id"] in seen:
            continue
        seen.add(d["id"])
        rows.append(d)
        if lim and len(rows) >= lim:
            break
    print("题目 %d" % len(rows), flush=True)

    out, t0 = {}, time.time()
    for i, d in enumerate(rows):
        out[d["id"]] = {b: keep_text(tok, d, float(b) / 100.0) for b in BUD}
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1fs  预计 %.0f 分钟"
                  % (i + 1, len(rows), el,
                     el / (i + 1) * (len(rows) - i - 1) / 60), flush=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False)
    print("done %d 题，用时 %.0f 秒" % (len(out), time.time() - t0))


if __name__ == "__main__":
    main()
