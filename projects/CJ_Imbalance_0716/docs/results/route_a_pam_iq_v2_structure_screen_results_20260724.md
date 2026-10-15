# 路线 A：PAM-IQ V2 双路径标签查询结构筛选结果（2026-07-24）

## 1. 结论先行

本轮在实验开始前冻结了网络结构、分类器、阈值、损失、超参数、训练过程和晋级门槛，随后完成了 `4 个结构 × 3 个随机种子 × 20 epochs` 的正式 development 试验。

最终结论如下：

1. **无注意力全局旁路有价值，但只改善了保守决策。** D1 的平均 Hamming loss 为 `0.2613`，优于 B1 的 `0.2724`，但弱分量召回仅为 `0.4393`，比 B1 低 `0.1168`。
2. **标签查询能从 D1 中恢复部分弱分量，但同时增加误报。** Q1 相对 D1 的弱分量召回提高 `0.0357`，但 Hamming loss 增加 `0.0194`。
3. **Q1 没有通过预注册晋级门槛。** 相对 B1，Q1 的 macro-AUPR、SawFM-AUPR、macro-F1、Hamming 和弱分量召回均未满足门槛，且弱召回在 0/3 seeds 上取得正增益。
4. **F2 阈值暴露了隐藏弱响应，但不能当作有效提升。** Q1 的 F2 弱召回达到 `0.9205`，但 Hamming 上升至 `0.3698`；SawFM、VFT、ISRJ 在该阈值下的负样本误报率约为 `0.9950/0.9817/0.9954`，实质上接近“几乎全报正”。
5. **按预注册规则，本轮没有继续执行第二阶段弱正排序损失。** 当前查询结构还没有形成可靠的弱正—负样本分离，直接加入弱正损失会混淆结构与目标函数归因。

因此，Q1 不能晋级为新的路线 A 主模型；B1/Base-DB 继续作为六类阶段主要对比基线。D1 和 Q1 保留为机制消融证据。

## 2. 冻结实验方案

完整预注册方案见：

`docs/implementation/route_a_pam_iq_v2_dual_query_experiment_plan_20260724.md`

### 2.1 输入与共享编码器

输入为 RMS 归一化复数 I/Q：`[B,2,3000]`。PAM-IQ 编码器输出时序特征：

```text
[B,2,3000]
  -> complex residual blocks + phase interaction fusion
  -> three multi-scale temporal stages
  -> H = [B,128,375]
```

V2 将 ECA、PCAF、GlobalContext 改为残差注意力：

```text
y = x + tanh(gamma) * (Attention(x) - x)
```

`gamma` 初始化为 `0.10`，避免连续乘法门控直接删除低响应证据。

### 2.2 D1：双路径统计池化

```text
global path: H -> unweighted mean/std -> Linear(256,128)
local path : H -> attentive mean/std  -> Linear(256,128)
fusion     : g + tanh(gamma_pool) * a
```

部署参数量：`286,481`。

### 2.3 Q1：双路径标签查询

全局路径与 D1 一致；局部路径由 6 个标签查询分别寻找局部证据：

```text
query_dim = 32
key/value = Conv1d(128,32,1)
attention_floor = 0.20
Z = [B,6,32]
```

分类器为全局主头加标签查询残差头：

```text
global_logits = Linear(128,6)(g)
query_logit[c] = dot(z[c], w[c]) + b[c]
logit[c] = global_logits[c] + tanh(gamma_query[c]) * query_logit[c]
```

`gamma_query[c]` 初始化为 `0.10`。Q1 部署参数量为 `257,819`，与 B1 的 `260,630` 基本匹配。

### 2.4 损失、阈值和训练参数

- 第一阶段损失：完整 DB Loss；不混入 ASL、弱正加权、功率配对或 prototype。
- 主要模型选择：development role 的 macro-AUPR。
- 主要决策阈值：shrinkage per-label F1。
- 附加阈值：fixed 0.5、global、per-label F1、per-label F2、per-label Youden。
- 优化器：AdamW，`lr=3e-4`，`weight_decay=1e-4`。
- `batch_size=64`，`epochs=20`，`patience=20`，warmup `10%`。
- `dropout=0.20`，gradient clip `5.0`，AMP 开启。
- 训练样本数 `2400`，development evaluation 每角色最多 `300`。
- seeds：`2041/2042/2043`。
- locked test：冻结，`locked_test_used=false`。

### 2.5 对照矩阵

| 编号 | 模型 | 部署参数量 | 作用 |
|---|---|---:|---|
| B1 | WideCapacity/Base-DB | 260,630 | 容量匹配主基线 |
| P0 | 原始 PAM-IQ | 253,325 | 原结构锚点 |
| D1 | PAM-IQ + 双路径统计池化 | 286,481 | 验证全局安全旁路 |
| Q1 | PAM-IQ + 双路径标签查询 | 257,819 | 验证标签条件局部证据 |

## 3. 实施与审计

实现文件：

- `cjlt/models/phase_multiscale.py`
- `cjlt/models/backbone_adapter.py`
- `cjlt/engine/thresholding.py`
- `experiments/run_tapc.py`
- `experiments/run_route_a_pam_iq_v2_structure_screen.py`
- `experiments/analyze_route_a_pam_iq_v2_scores.py`

验证结果：

- 定向单元测试：`21 passed`。
- 全量单元测试：`143 passed, 36 warnings`。
- GPU smoke：4/4 结构成功完成。
- 正式试验：12/12 runs 成功完成。
- 同 seed 的四个结构共享相同 split SHA256 和 manifest SHA256。
- 所有输出均标记 `locked_test_used=false`。

正式输出：

```text
outputs/tapc/route_a_pam_iq_v2_structure_screen/
  route_a_pam_iq_v2_structure_screen_summary.json
  route_a_pam_iq_v2_score_diagnostics.json
  runs/<model>/seed_<seed>/
```

## 4. 正式结果

下表为三随机种子均值 ± 样本标准差。

| 模型 | macro-AUPR | SawFM-AUPR | macro-F1 | Hamming ↓ | 弱分量召回 | F2 弱召回 | forgetting ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.8481 ± 0.0048 | 0.7486 ± 0.0149 | 0.7582 ± 0.0056 | 0.2724 ± 0.0099 | 0.5561 ± 0.0127 | 0.8759 ± 0.0614 | 0.0816 ± 0.0156 |
| P0 | 0.8363 ± 0.0111 | **0.7789 ± 0.0387** | 0.7136 ± 0.0134 | 0.2846 ± 0.0184 | 0.4278 ± 0.0580 | **0.9441 ± 0.0225** | 0.0478 ± 0.0437 |
| D1 | 0.8410 ± 0.0047 | 0.7272 ± 0.0479 | 0.7395 ± 0.0134 | **0.2613 ± 0.0104** | 0.4393 ± 0.0508 | 0.8857 ± 0.0258 | 0.0607 ± 0.0183 |
| Q1 | 0.8385 ± 0.0070 | 0.7307 ± 0.0179 | 0.7329 ± 0.0097 | 0.2807 ± 0.0234 | 0.4750 ± 0.0397 | 0.9205 ± 0.0325 | **0.0352 ± 0.0161** |

最佳 epoch：

- B1：`7/13/13`
- P0：`5/11/10`
- D1：`20/13/17`
- Q1：`15/20/5`

Q1 的最佳 epoch 波动明显大于 B1，说明新查询分支仍存在优化不稳定性。

## 5. 关键配对差值

### 5.1 Q1 相对 B1

| 指标 | 平均差值 |
|---|---:|
| macro-AUPR | -0.0096 |
| SawFM-AUPR | -0.0179 |
| macro-F1 | -0.0253 |
| Hamming loss | +0.0083 |
| 弱分量召回 | -0.0811 |
| F2 macro-F1 | -0.0016 |
| F2 Hamming loss | +0.0126 |
| F2 弱分量召回 | +0.0446 |
| forgetting | -0.0463 |

逐标签 AUPR 差值：FMB `+0.0018`、SawFM `-0.0179`、SinFM `-0.0436`、MFT `-0.0230`、VFT `-0.0321`、ISRJ `+0.0572`。

这说明 Q1 对 ISRJ 有稳定收益，并基本保持 FMB，但损害了 SawFM、SinFM、MFT 和 VFT，仍然是标签间收益转移，而不是六标签共同提升。

### 5.2 Q1 相对 P0

| 指标 | 平均差值 |
|---|---:|
| macro-AUPR | +0.0023 |
| macro-F1 | +0.0192 |
| Hamming loss | -0.0039 |
| 弱分量召回 | +0.0473 |
| SawFM-AUPR | -0.0482 |

Q1 修复了 P0 的一部分通用分类和弱召回问题，但同时破坏了 P0 对 SawFM 的尾类偏置。因此，不能把 Q1 视为 P0 的全面升级。

### 5.3 Q1 相对 D1

| 指标 | 平均差值 |
|---|---:|
| macro-AUPR | -0.0024 |
| SawFM-AUPR | +0.0035 |
| macro-F1 | -0.0067 |
| Hamming loss | +0.0194 |
| 弱分量召回 | +0.0357 |
| F2 弱召回 | +0.0347 |

标签查询分支确实带来了额外检测，但这些检测没有足够的精度约束。

## 6. 门控与分数分布诊断

### 6.1 学到的残差门控没有爆炸

- D1 三层时间注意力的 `tanh(gamma)` 约为 `0.077–0.142`。
- D1 池化残差的 `tanh(gamma_pool)` 约为 `0.160–0.203`。
- Q1 标签查询残差的 `tanh(gamma_query)` 约为 `0.100–0.188`，均值 `0.132`。
- Q1 查询分类器的有效 L2 范数约为 `0.120–0.290`，均值 `0.181`。

因此，Q1 的退化不能归因于查询门控幅值异常增大。残差形式达到了“不会一开始覆盖全局主头”的设计目的，但尚未学出足够判别性的标签局部证据。

### 6.2 弱正与负样本仍高度重叠

| 模型 | 弱正均值 | 弱正中位数 | 强正中位数 | 负样本中位数 | 弱正-vs-负 AUROC | 弱正-vs-负 AUPR |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.4375 | 0.4175 | 0.6351 | 0.3527 | 0.6076 | 0.3014 |
| P0 | 0.4250 | 0.3438 | 0.7416 | 0.3091 | 0.5677 | 0.2815 |
| D1 | 0.4328 | 0.3401 | 0.7489 | 0.2922 | 0.5917 | 0.3242 |
| Q1 | 0.4304 | 0.3660 | 0.7267 | 0.3140 | 0.5845 | 0.3140 |

Q1 虽将弱正中位数从 D1 的 `0.3401` 抬到 `0.3660`，但负样本中位数也从 `0.2922` 抬到 `0.3140`，使弱正—负样本 AUROC 从 `0.5917` 降至 `0.5845`。这正是“召回增加、误报同步增加”的直接证据。

### 6.3 F2 阈值的高召回主要来自过低阈值

| 模型 | 主阈值负样本 FPR | F2 阈值负样本 FPR |
|---|---:|---:|
| B1 | 0.3556 | 0.6880 |
| P0 | 0.2712 | 0.8174 |
| D1 | **0.2419** | 0.7170 |
| Q1 | 0.3096 | 0.7381 |

Q1 逐标签诊断中：

- FMB 弱正-vs-负 AUROC `0.8157`，说明 FMB 的弱证据具备一定可分性。
- SinFM AUROC `0.7310`，也存在可利用的弱证据。
- SawFM AUROC 仅 `0.4268`，低于随机水平；其 F2 弱召回 `0.9912` 对应负样本 FPR `0.9950`。
- VFT 的 F2 弱召回 `0.9813` 对应负样本 FPR `0.9817`。
- ISRJ 的 F2 弱召回 `1.0000` 对应负样本 FPR `0.9954`。

所以，当前不能用 F2 弱召回单独证明弱分量识别改善。必须同时报告弱正—负样本 AUROC/AUPR、Hamming、逐标签 FPR 或 precision。

## 7. 失败原因归因

### 7.1 已验证的原因

1. **全局安全旁路偏保守。** D1 降低总体误报，但无法主动寻找弱局部证据。
2. **标签查询缺少弱正—负样本判别目标。** DB Loss 仍只处理标签频次，查询分支主要受到强正样本和普通负样本支配。
3. **同一查询残差同时抬高弱正和负样本。** Q1 改善部分召回，却没有提高整体弱正—负样本排序能力。
4. **不同标签的问题不同。** FMB/SinFM 已有可分弱证据，SawFM 弱证据反而低于负样本；统一阈值目标或统一查询约束不合适。
5. **P0 的 SawFM 收益与通用可分性存在冲突。** V2 提高通用指标时丢失 SawFM 尾类偏置，说明需要保留或蒸馏 P0 的 SawFM 特征，而不是完全替换其池化头。

### 7.2 被排除或削弱的解释

- 不是简单参数量不足：Q1 与 B1 容量基本匹配。
- 不是残差门控爆炸：学习后的门控仍处于小幅残差范围。
- 不是单纯阈值不合适：即使改变为 F2，弱召回的增加伴随极高负样本误报；排序本身仍未改善。
- 不是继续堆叠卷积尺度即可解决：共享编码器能为 FMB/SinFM 提供弱证据，但 SawFM 等标签缺少正确局部判别机制。

## 8. 预注册晋级门槛判定

Q1 相对 B1 的六项检查全部失败：

| 检查 | 结果 |
|---|---|
| macro-AUPR 非劣 `>= -0.005` | 失败，实际 `-0.0096` |
| SawFM-AUPR 增益 `>= +0.010` | 失败，实际 `-0.0179` |
| 弱召回增益 `>= +0.030` | 失败，实际 `-0.0811` |
| Hamming 非劣 `<= +0.005` | 失败，实际 `+0.0083` |
| 保护标签下降不超过 0.030 | 失败，SinFM/VFT 超限 |
| 弱召回至少 2/3 seeds 正增益 | 失败，实际 0/3 |

自动决策：

```text
revise_or_stop_q1_before_weak_visibility_objective
```

## 9. 下一版结构整改方向

本轮不直接执行以下方案；它们是由正式证据推导出的下一轮候选。

### 9.1 将 D1 作为主干，查询头改为受约束专家

保留 D1 的低误报全局主头，查询分支不再无条件相加，而采用置信门控：

```text
logit[c] = global_logit[c]
         + gate[c] * clipped_query_residual[c]
```

建议：

- `gate[c]` 同时读取全局特征、查询特征和二者一致性；
- 查询残差限制在固定范围，例如 `tanh(residual) * 0.5`；
- 加入 query-residual L2 和负样本激活惩罚；
- 先冻结 D1，仅训练查询专家，避免破坏全局低误报边界。

### 9.2 按标签分流，而不是强制六标签共享同一种整改

- FMB/SinFM：已有较高弱正—负 AUROC，可优先尝试弱正排序和精度约束。
- SawFM：当前弱正—负 AUROC 低于随机，必须先验证标签条件局部区域是否正确；优先做 oracle-positive attention/localization 和 P0-SawFM 蒸馏。
- MFT/VFT/ISRJ：先限制负样本查询激活，再评估是否存在可靠弱证据。

### 9.3 第二阶段损失必须同时约束弱召回和误报

下一版不能只使用弱正—负排序，应采用：

```text
L = L_DB
  + lambda_rank * L_weak_positive_vs_hard_negative
  + lambda_fp   * L_query_negative_suppression
  + lambda_distill * L_P0_SawFM_or_D1_global_distillation
```

并将晋级指标改为联合约束：

- 弱正—负样本 AUROC/AUPR 必须提高；
- weak recall 提高时，逐标签负样本 FPR 不得显著恶化；
- macro-AUPR、SawFM-AUPR 和 Hamming 保持非劣；
- 不再把低阈值下的单独弱召回作为有效证据。

## 10. 证据边界

本轮结果属于冻结 locked test 之前的 development 结构筛选。它证明了：

- 全局旁路可降低误报；
- 标签查询可增加检测，但当前没有形成可靠弱正—负分离；
- 阈值可以放大弱召回，却不能替代表示和分类边界证据。

本轮没有证明 Q1 优于 Base-DB，也没有授权读取 locked test。后续只有在新结构通过预注册 development 门槛后，才应进入更长种子验证与最终冻结测试。
