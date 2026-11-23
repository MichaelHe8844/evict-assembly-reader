# 论文表格 → 结果文件索引

记录每张表的原始文件与字段，便于复核。
字段说明：`records` + `correct`（arm_c 系列）；`detail` + `ok`（oracle_ceiling 系列）；`summary`（汇总）。

| 论文位置 | 结果文件 | 数据集 | 复核状态 |
|---|---|---|---|
| 表 2 先数值后绑定 | `vf_af3/5/10/20.json`（各档一文件） | dev 126 | ✅ 0.270/0.294/0.333/0.325；对旧顺序 +28/−3、+18/−1、+6/−3、+1/−0，p 与正文一致 |
| 表 2 旧顺序（表头优先） | `e2e_af{3,5,10,20}_base.json` | dev 126 | ✅ 0.071/0.159/0.310/0.317 |
| 表 2 只保数字 | `numb_tight.json`（5/10/20%）、`nf_af3.json`（3%） | dev 126 | ✅ 0.230/0.286/0.341/0.349 |
| 表 3（LM $\mathcal{A}_0\to\mathcal{A}_1$） | `vf216_new.json`、`vf216_sel_new.json` | test 216 | ✅ 0.116→0.407，翻转 +66/−3 |
| 表 4（两个上界） | 无数值结果文件；数据见 `预实验报告_数值锚点KV压缩.md` §3.42 的表 | dev 87 | ✅ 0.402/0.506/0.529、0.345/0.379/0.379、85.7/75.0/71.7% 逐项一致；绘图脚本 `plot_ceiling.py` 为硬编码副本 |
| 表 5 程序（gold / EFST） | `vf216_gold_vf.json`、`vf216_rft_vf.json`（detail/base|force） | test 216 | ✅ 逐档一致 |
| 表 5 LM 两列 | 同表 3 | test 216 | ✅ |
| 5.5 八个区间 | 上述四个文件 | test 216 | ✅ 与重算一致（±0.2 bootstrap 噪声） |
| 表 6 提示列 | `x3_base.json`、`x5_base.json`、`x10_base.json`（182） | 182 子集 | ✅ 0.137/0.280/0.407 |
| 表 6 直接答题 / $\mathcal{A}_0$·程序 | `crossbase_test.json`（qwen/llama · rows） | 182 子集 | ✅ 逐档一致 |
| 表 6 $\mathcal{A}_1$·程序 | `locator_ceiling_bc_test.json`（force） | 182 子集 | ✅ 0.462/0.516/0.560/0.665 |
| 表 7 多步（221） | `ms_reader.json`（226，base）∩ `ms_lm226.json`（221） | test 221 | ✅ 交集重算得 0.100/0.249/0.421/0.548 |
| 表 8 SnapKV+钉对 / +随机对 | `cross216_merged.json`（pin / rand） | test 216 | ✅ 0.361…/0.014… |
| 表 8 H2O+钉对 | `_h2opin_c{0-3}.json` 合并 | test 216 | ✅ 0.370/0.347/0.356/0.338 |
| 表 8 纯压缩基线 | `cross216_merged.json`（pure）与 `cross_pure216.json` | test 216 | ⚠️ 两文件冲突（0.009… vs 0.019/0.023…） |
| 表 9 三条件 | `_tkv_r{0-3}.json`、`_tkv2_r{0-3}.json`、`_a0_r{0-3}.json` | test 216 | ✅ |
| 5.11 其三 纯钉对 | `_pv_noforce{0-3}.json`、`_pv_force{0-3}.json` | test 216 | ✅ +12.0/+6.9/+2.8/+6.5 |
| 5.3 三臂（182，5%） | `vf_x5.json`、`vf_x5_rule.json`、`vf_x5_sel.json` | 182 子集 | ✅ +24/−11，p=0.041 |

## 未闭合项

1. **表 8 纯压缩基线有两个版本**：`cross216_merged.pure`（0.009/0.005/0.005/0.009，与同文件的 pin/rand 同源）与 `cross_pure216.snap`（0.019/0.023/0.005/0.037，另一次纯基线重测）。表 8 采用前者——它与 pin 行同一次运行，增量（0.361−0.009=35.2）因此自洽；若改用后者，增量会变成 30–34，且属于跨运行相减，不建议。
2. **表 4**：无逐题 JSON，只有报告 §3.42 的汇总表；如需逐题复现，须重跑 87 道 dev 的「两数是否同时保留」检查。
3. `cross_pin216.json.partial.json` 只有 85/216（运行中断），不可作为 216 题结果的依据。

## 复现注意

1. `arm_c.py` 的 `VALUE_FIRST` 默认 **False**；$\mathcal{A}_1$ 需显式 `--value_first`。
2. `oracle_ceiling.py` 的 force 分支按 `KIND_PRIO`（value 优先）重排剩余项，故 base→force 同时改变**顺序与钉对**。
3. 结果文件有两套 schema（`records`/`detail`），检索时需同时检查。
4. 正文配对增量/CI/翻转按未舍入值计算，与表内三位小数直接相减会差 0.1pp。
5. 若出现 `*.json.partial.json`，说明该次运行中断，记录不完整。
