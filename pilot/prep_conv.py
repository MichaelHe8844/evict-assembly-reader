# -*- coding: utf-8 -*-
"""把 ConvFinQA 的 dev_turn 预处理成 FinQA 同构输入。

关键一步：把**本轮之前的对话**拼进 pre_text。ConvFinQA 的后续轮次大量使用
"this / that / the same period"，不带上文就无法作答；而这些指代正是「绑定」
最吃紧的场景，也正是这个数据集值得跑的原因。

输出可直接作为 arm_c --data 的输入（内部仍走 common.build_dataset）。
"""
import json
import sys


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "../data/conv/data/dev_turn.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "../results/conv_dev.json"
    d = json.load(open(src, encoding="utf-8"))
    out, n_hist, skipped = [], 0, 0
    for ex in d:
        if not isinstance(ex.get("qa"), dict):
            skipped += 1
            continue
        dial = (ex.get("annotation") or {}).get("cur_dial") or []
        hist = dial[:-1] if len(dial) > 1 else []
        if hist:
            n_hist += 1
        pre = list(ex.get("pre_text") or [])
        if hist:
            pre = ["Conversation so far: " + " ".join(hist)] + pre
        out.append(dict(id=ex["id"], filename=ex.get("filename"),
                        pre_text=pre, post_text=ex.get("post_text") or [],
                        table=ex.get("table") or [], qa=ex["qa"],
                        hist_len=len(hist)))
    json.dump(out, open(dst, "w"), ensure_ascii=False)
    print("写出 %d 条，其中带上文 %d（%.0f%%），跳过无 qa %d"
          % (len(out), n_hist, 100 * n_hist / max(1, len(out)), skipped))


if __name__ == "__main__":
    main()
