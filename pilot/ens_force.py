"""B5：多种子集成的选择器。

不做新训练：把 r1c / r1_s1 / r1_s2 / r1_s3 四个检查点的 logit 平均，
再取 argmax 对，输出与 arm_c 的 --force_json 兼容的文件。
"""
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from r1_feat import QDS, collate, M

BASE = "E:/KV_models/bge-small-en-v1.5"
CKPTS = ["r1c.pt", "r1_s1.pt", "r1_s2.pt", "r1_s3.pt"]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = M(BASE).cuda()
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    models = [load("../results/" + c) for c in CKPTS]

    for name, path, out in (("dev", "../results/r1_pairs_dev_full.json", "ens_force_dev"),
                            ("test", "../results/xds_pairs_test.json", "ens_force_test")):
        rows = json.load(open(path, encoding="utf-8"))
        ds = QDS(rows, tok, train=False)
        dl = DataLoader(ds, batch_size=16, shuffle=False,
                        collate_fn=lambda b: collate(b, pad))
        picks, agree, idx = {}, 0, 0
        with torch.no_grad():
            for ids, am, pm, y, F in dl:
                zs = [m(ids.cuda(), am.cuda(), F.cuda()).float().cpu() for m in models]
                s = (zs[0] if len(zs) == 1 else sum(zs[1:], zs[0])).masked_fill(~pm, -1e4)
                k = s.argmax(1)
                singles = [z.masked_fill(~pm, -1e4).argmax(1) for z in zs]
                for i in range(s.shape[0]):
                    row = rows[idx]
                    j = int(k[i])
                    agree += all(int(t[i]) == j for t in singles)
                    a, b, _ = row["pairs"][j]
                    ca, cb = row["cands"][a], row["cands"][b]
                    pid = row["key"][0] if isinstance(row["key"], (list, tuple)) else row["key"]
                    picks[pid] = [[ca["r"], ca["c"]], [cb["r"], cb["c"]]]
                    idx += 1
        json.dump(picks, open("../results/%s.json" % out, "w", encoding="utf-8"),
                  ensure_ascii=False)
        print("%s：%d 条，四模型完全一致 %d 条（%.1f%%）"
              % (name, len(picks), agree, 100 * agree / max(1, len(picks))), flush=True)


if __name__ == "__main__":
    main()
