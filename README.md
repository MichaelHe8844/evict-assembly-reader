# 压缩证据下的取证装配与编译式读取

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24459338.svg)](https://doi.org/10.5281/zenodo.24459338)


本仓库是论文《压缩证据下的取证装配与编译式读取》的代码与逐题结果。核心主张：在 KV 缓存被压到 3%–20% 时，数值问答的瓶颈不在"打分够不够准"，而在**保留单位**——数值能否和它的行头、列名、单位一起留下；以及**读取方式**——算术交给确定性执行器，而不是让模型口算。

## 目录

| 目录 | 内容 |
|---|---|
| `pilot/` | 实验脚本。主入口 `arm_c.py`（装配 + 读取的一次完整评测），`oracle_ceiling.py`（文本通道上界） |
| `results/` | 逐题结果 JSON（511 个，74 MB）。`MANIFEST.md` 列出每个文件的顶层字段 |
| `_tpl/` | 论文排版脚本（Markdown → docx），与实验无关，可忽略 |
| `图1_三层方法.drawio` | 图 1 的 draw.io 源文件 |
| `*.sh` | 批量驱动脚本，展示实际使用的参数组合 |

## 环境

- Python 3.10+，PyTorch 2.x，`transformers`、`peft`、`bitsandbytes`、`datasets`、`scipy`
- 单卡 16 GB 可跑 3B + 4-bit nf4 + LoRA（本文实验为 RTX 4060 Ti 16 GB）
- `pip install -r requirements.txt`

## 数据

FinQA 与 ConvFinQA 为公开数据集，**不随仓库分发**（遵循原数据集许可）。请自行下载后放到 `data/`：

```
data/train.json      # FinQA 官方 train
data/dev.json        # FinQA 官方 dev
data/test.json       # FinQA 官方 test（含 conv 变体时另放 conv_test.json）
```

评测集构造规则见论文 5.1：从官方 test 中筛出单步、可解析、文档长度 64–1100 token 的 216 题。

## 路径配置

脚本内有几处写死的本机路径，运行前请改成你的环境：

- `pilot/arm_c.py` 的 `--model` 默认值 `E:/KV_models/Qwen2.5-3B-Instruct`
- `*.sh` 里的 `PY=/e/anaconda3/envs/paper/python.exe`

建议改用命令行参数覆盖，而不是改源码：

```bash
python pilot/arm_c.py --model /path/to/Qwen2.5-3B-Instruct ...
```

## 复现主线

三条主线分别对应论文的表 4、表 3、表 5。公共参数：`--task finqa_pot`（先出算式、由脚本求值）、`--single_step`（只取单步题）、`--budget_mode fixed`（逐样本同预算）。

**① 保留粒度对照（表 4）**：同一注意力信号下，token 级 top-B 与组级整装对比。

```bash
python pilot/arm_c.py --data data/test.json --task finqa_pot --single_step \
  --limit 216 --load_4bit --value_first --out results/表4_AUA.json
```

**② 读者 × 装配（表 3）**：`--adapter` 指定编译式读者的 LoRA 权重；不传则走同底座未微调的语言模型读者。

```bash
python pilot/arm_c.py --data data/test.json --task finqa_pot --single_step \
  --limit 216 --load_4bit --adapter /path/to/lora --value_first --out results/表3_prog_A1.json
```

**③ 跨底座（表 5）**：换 `--model` 为 `Llama-3.2-3B-Instruct`，其余不变；评测集为 182 题（剔除 20% 档无对可钉的 34 题）。

```bash
python pilot/arm_c.py --model /path/to/Llama-3.2-3B-Instruct --data data/test.json \
  --task finqa_pot --single_step --limit 182 --load_4bit --value_first \
  --out results/表5_llama.json
```

其余开关（`--min_bind` 钉对、`--no_attn` 关闭注意力填充、`--zero_bind_layers` 干预实验等）见 `python pilot/arm_c.py --help`。逐题结果写出后，配对增量与聚类 bootstrap 区间由 `pilot/analyze*.py` 系列脚本计算。

## 结果文件

`results/MANIFEST.md` 逐条列出文件名、体积与顶层字段。命名约定：

- `arm_c_*`：`arm_c.py` 的输出，文件名即运行条件
- `vf216_*`：216 题主表相关
- `ms_*`：多步题（221 题）
- `oos_*`：留出集

体积较大的数据集副本与上下文缓存（`conv_*`、`ctx_cache_*`、`*_rows`）未纳入仓库，可按需索取。

## 许可

代码以 MIT 许可发布，见 `LICENSE`。数据与模型权重遵循各自原始许可（FinQA/ConvFinQA、Qwen、Llama）。

## 联系

通信作者邮箱见论文首页。复现遇到问题请开 issue。
