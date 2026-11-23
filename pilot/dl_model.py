# -*- coding: utf-8 -*-
"""从 ModelScope 拉模型（HF 在本机被墙）。只取 transformers 需要的文件，跳过 original/*.pth。"""
import sys
from modelscope import snapshot_download

mid = sys.argv[1]
out = sys.argv[2]
p = snapshot_download(
    mid,
    local_dir=out,
    ignore_file_pattern=[r"original/.*", r".*\.pth", r".*\.md", r".*\.pdf"],
)
print("OK", p)
