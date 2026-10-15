# 路线 A：Head-DB 六类扩展筛选结果（2026-07-23）

## 1. 结论

六类扩展已经完成，但 **Head-DB 未通过预注册的 3-seed 晋级门槛**，因此本轮不扩展到 5 个随机种子，也不使用 locked test。

结果不是“六类完全失败”，而是出现了明确的目标分离：

- Head-DB 相对 Base-ASL 的整体判别能力提高：macro-AUPR `+0.008763`，95% CI `[+0.005399,+0.012086]`，18 个配对中 15 个获胜；
- macro-F1 提高 `+0.005162`，Hamming loss 改善 `-0.006605`；
- 但目标尾类 AUPR 的总体变化仅为 `-0.000075`，95% CI `[-0.005202,+0.004691]`，未形成稳定尾类恢复；
- 相对端到端 Base-DB，Head-DB 的 macro-AUPR 和 tail-AUPR 分别低 `0.009106` 和 `0.022692`。

因此，四类实验中成立的“冻结 ASL 表征，仅用 DB 重训分类头”不能原样推广为六类最终方法。六类下的主要瓶颈已经从分类边界校正转向有限的表征适配。

## 2. 冻结的实验协议

| 项目 | 设置 |
|---|---|
| 配置 | `configs/student_a/six_class.yaml` |
| 数据来源 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| Python 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 类别顺序 | FMB、SawFM、SinFM、MFT、VFT、ISRJ |
| 协议 | controlled Label-LT |
| 总训练样本数 | 2400，精确约束 |
| 尾类正样本锚点 | 400 |
| 不平衡因子 | IF=4 |
| 目标正样本数 | 约 `[1600,1213,919,696,528,400]`，随尾类轮换 |
| 随机种子 | 2041、2042、2043 |
| 正式运行数 | 6 个尾类 × 3 seeds × 3 方法 = 54 |
| Base-ASL / Base-DB | 20 epochs，端到端训练 |
| Head-DB | 从对应 Base-ASL 初始化，冻结表征，10 epochs，lr=1e-3 |
| 模型选择 | optimization role 的 macro-AUPR |
| locked test | 未使用 |

三种方法在同一尾类与 seed 下共享完全相同的 split SHA 和 manifest SHA。所有 manifest 的样本数均精确为 2400，标签计数最大误差为 0。

## 3. Head-DB 相对 Base-ASL 的逐尾类结果

| 目标尾类 | Δ macro-AUPR | 95% CI | Δ tail-AUPR | 95% CI | macro 获胜 | tail 获胜 |
|---|---:|---:|---:|---:|---:|---:|
| FMB | +0.016732 | [+0.012456,+0.022499] | +0.009016 | [+0.004645,+0.013840] | 3/3 | 3/3 |
| SawFM | +0.008664 | [+0.005099,+0.010701] | -0.011007 | [-0.025512,-0.000363] | 3/3 | 0/3 |
| SinFM | +0.005265 | [+0.002836,+0.009930] | +0.008086 | [+0.006826,+0.010238] | 3/3 | 3/3 |
| MFT | +0.013584 | [+0.005017,+0.020424] | +0.008242 | [+0.005067,+0.011637] | 3/3 | 3/3 |
| VFT | +0.010791 | [+0.007427,+0.013435] | -0.003107 | [-0.015540,+0.003896] | 3/3 | 2/3 |
| ISRJ | -0.002455 | [-0.004236,-0.001411] | -0.011683 | [-0.012294,-0.011286] | 0/3 | 0/3 |

总体结果：

| 指标 | 均值变化 | bootstrap 95% CI |
|---|---:|---:|
| macro-AUPR | +0.008763 | [+0.005399,+0.012086] |
| tail-AUPR | -0.000075 | [-0.005202,+0.004691] |
| macro-F1 | +0.005162 | [+0.001881,+0.008419] |
| Hamming loss | -0.006605 | [-0.011296,-0.002407] |
| tail-F1 | +0.007847 | [-0.002338,+0.018449] |

## 4. Head-DB 相对端到端 Base-DB

总体上，Head-DB 相对 Base-DB：

| 指标 | 均值变化 | bootstrap 95% CI |
|---|---:|---:|
| macro-AUPR | -0.009106 | [-0.012574,-0.005435] |
| tail-AUPR | -0.022692 | [-0.036605,-0.008913] |
| macro-F1 | +0.004062 | [+0.001403,+0.006372] |
| Hamming loss | -0.000802 | [-0.004568,+0.002963] |

目标尾类 AUPR 的实际均值为：

| 尾类 | Base-ASL | Base-DB | Head-DB(10 epoch) |
|---|---:|---:|---:|
| FMB | 0.939611 | 0.941094 | 0.948627 |
| SawFM | 0.712712 | 0.736400 | 0.701705 |
| SinFM | 0.934578 | 0.931637 | 0.942664 |
| MFT | 0.888580 | 0.904597 | 0.896822 |
| VFT | 0.621006 | 0.668724 | 0.617899 |
| ISRJ | 0.661321 | 0.711053 | 0.649638 |

SawFM、VFT、ISRJ 在 Base-DB 下明显受益于表征共同更新，而仅调整 390 个分类头参数无法恢复该收益。

## 5. 20-epoch 排除诊断

由于 SawFM、VFT 的 10-epoch Head-DB 最优轮次经常位于训练末端，追加了 3 个问题尾类 × 3 seeds 的 Head-DB 20-epoch 诊断，共 9 组；初始检查点、数据清单和其他参数均保持不变。

| 尾类 | Head20-ASL Δ macro-AUPR | Head20-ASL Δ tail-AUPR | Head20-Head10 Δ tail-AUPR | Head20-BaseDB Δ tail-AUPR |
|---|---:|---:|---:|---:|
| SawFM | +0.010132 | -0.011873 | -0.000866 | -0.035562 |
| VFT | +0.013142 | +0.003093 | +0.006200 | -0.044624 |
| ISRJ | -0.002452 | -0.011807 | -0.000124 | -0.061540 |

诊断结论：

- VFT 的确有一部分是训练预算不足，20 epochs 后可从相对 ASL 的 `-0.003107` 改为 `+0.003093`；
- SawFM 和 ISRJ 延长训练后没有恢复；
- 三类相对 Base-DB 仍有大幅差距，因此轮数不是主要原因；
- 9 个诊断运行仍然只有 390 个可训练参数，所有非分类头状态逐元素不变，locked test 均为 false。

## 6. 失败原因分析

### 6.1 不是阈值问题

失败首先体现在 tail-AUPR。AUPR 基于预测排序、与 0.5 阈值和校准阈值无关，因此不能通过重新选阈值解释或修复 SawFM/ISRJ 的下降。

### 6.2 DB 分类头改善了整体，但没有保证目标尾类

Head-DB 在 18 个配对中有 15 个 macro-AUPR 提升，却只有 11 个 tail-AUPR 提升。对 SawFM 尾类，宏指标的主要收益来自 VFT 等非目标标签；对 VFT 尾类，收益主要来自 FMB、SawFM 等标签。当前 DB 头优化的是六个输出的总损失，不包含“当前目标尾类必须优先恢复”的约束。

### 6.3 六类需要有限的表征适配

Base-DB 对 SawFM、VFT、ISRJ 的尾类 AUPR 分别达到 0.736400、0.668724、0.711053，均显著高于冻结头方案。这说明 DB 信号本身有效，但这些类别需要改变共享时序表征，而不仅是移动最后的线性决策面。

### 6.4 新增类别改变了类间竞争关系

四类阶段 ISRJ 的 Head-DB 总体为轻微正向；加入 SawFM、VFT 后，ISRJ 在三个 seeds 上 tail-AUPR 均稳定下降约 0.011–0.012。该翻转说明六类任务不是简单增加两个独立输出，新增调制类型改变了共享表征中的混淆与标签共现竞争。

### 6.5 训练预算只是次要因素

20-epoch 诊断只修复了 VFT 相对 Base-ASL 的非劣性，没有修复 SawFM、ISRJ，也没有追平 Base-DB。因此下一步不应继续单独增加 Head-DB 轮数。

## 7. 下一阶段建议

下一阶段应保留“两阶段训练”和 locked-test 治理，但把 Stage 2 从“仅分类头”升级为“受约束的局部表征适配”：

1. 从六类 Base-ASL 初始化；
2. 冻结 complex stem、前两个 temporal blocks 和全部无关辅助分支；
3. 只解冻最后一个时序块 `backbone.temporal_backbone.5` 与分类头；
4. 分类头使用较大学习率，最后时序块使用 10 倍更小的学习率；
5. 使用 DB 尾类项与 ASL 非尾类保持项组成的标签条件化混合损失，避免宏指标收益被容易的非目标标签吸收；
6. 对初始 ASL 特征或非尾类 logits 加保持约束，限制局部适配造成的表征遗忘；
7. 第一轮只做 SawFM、VFT、ISRJ 三尾类 × 3 seeds 的机制筛选，并同时保留 Head-DB20 与 Base-DB 两个对照；
8. 只有 SawFM、ISRJ tail-AUPR 恢复且 macro-AUPR 非劣时，才扩展至六尾类和 5 seeds。

推荐的首轮消融顺序为：

- A6-PU：最后时序块 + 分类头，DB，判定“局部解冻是否足够”；
- A6-PU-SP：A6-PU + 起点参数保持约束，判定“是否需要抑制遗忘”；
- A6-TailMix：A6-PU-SP + 尾类 DB / 非尾类 ASL 条件化损失，判定“是否能把宏收益转化为尾类收益”。

## 8. 审计与产物

- 正式汇总：`outputs/tapc/route_a_six_class_head_db_screen/route_a_six_class_head_db_screen_summary.json`
- 54 组正式运行：`outputs/tapc/route_a_six_class_head_db_screen/runs/`
- 9 组轮数诊断：`outputs/tapc/route_a_six_class_head_db_epoch20_diagnostic/runs/`
- 冒烟运行：`outputs/tapc/route_a_six_class_head_db_smoke/`
- 技术方案：`docs/implementation/route_a_six_class_head_db_screen_plan_20260723.md`
- 批处理程序：`experiments/run_route_a_six_class_head_db_screen.py`

正式运行共 54 份 metrics、54 份 receipt 和 54 份 manifest；18 个 Head-DB 检查点均只有 390 个可训练参数，非分类头状态无漂移；所有正式与诊断运行均未使用 locked test。

正式汇总文件 SHA256：`7eb1528e238fdf9046bafbeec8c8fb35ba29832b72cf91cbb6d7678c5b691dec`。全量测试结果为 `118 passed`。
