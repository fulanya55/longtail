# 路线 A：SawFM 固定权重融合独立确认结果（2026-07-23）

## 1. 结论

本轮按长线验证要求，将发现阶段确定的超参数冻结后，使用全新 seeds 2039、2040 进行独立确认。**WeightFusion-0.2 未通过预注册确认门槛，静态参数融合应停止作为路线 A 的主候选。**

失败并不是宏观性能崩溃：两个新种子的 macro-AUPR 都保持在相对 Base-DB-Macro 的 `-0.005` 非劣界内；真正的问题是 SawFM 排序收益不能跨种子稳定复现，只有 seed 2040 为正，seed 2039 为负。发现阶段的 3/3 正收益因此应解释为候选发现证据，而不能解释为已经确认的普遍收益。

LogitFusion-0.6 在两个确认种子上均提高 SawFM AUPR，但持续降低 macro-F1、提高 Hamming loss，并且推理需要两个完整模型。它仍可作为“早期和后期端点存在输出互补性”的机制上界，不应作为部署主方法。

Base-DB-Macro 继续作为六类阶段的主要对比基线；locked test 继续冻结。

## 2. 本轮方案

| 项目 | 固定设置 |
|---|---|
| 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 数据 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 新确认 seeds | 2039、2040 |
| 训练样本 | 每 seed 2400，Label-LT，IF=4，SawFM 尾类锚点=400 |
| 训练 | 随机初始化、完整六标签监督、DB Loss、20 epochs |
| 后期端点 | DB-Macro，按 macro-AUPR 选点 |
| 早期端点 | DB-Tail，按 SawFM AUPR 选点 |
| WeightFusion | 固定 `alpha=0.2`，单模型参数插值 |
| LogitFusion | 固定 `alpha=0.6`，双模型输出融合 |
| 校准/验证 | 每 seed 各 300 样本，sample ID 交集为 0 |
| 阈值 | 只报告温度校准后的固定 0.5；未在验证角色选阈值 |
| locked test | 未使用 |

参数融合定义为：

```text
theta_fused = 0.8 * theta_macro + 0.2 * theta_tail
```

本轮没有重新搜索 alpha。融合脚本的 `selection_mode` 为 `fixed_from_prior_discovery`，只使用发现阶段冻结的 Weight `0.2` 与 Logit `0.6`。

## 3. 预注册确认门槛

WeightFusion 只有同时满足以下条件才可进入 VFT/ISRJ 跨类长线验证：

1. 两个确认种子的平均 `delta macro-AUPR >= -0.005`；
2. 两个确认种子的平均 `delta SawFM AUPR >= +0.005`；
3. macro-AUPR 非劣种子数为 2/2；
4. SawFM AUPR 正收益种子数为 2/2。

该门槛只根据 seeds 2039、2040 判断。2041—2043 与确认种子合并后的五种子结果只作描述性统计，不能替代独立确认。

## 4. 独立确认结果

### 4.1 WeightFusion-0.2

| Seed | DB-Macro macro-AUPR | DB-Macro SawFM AUPR | delta macro-AUPR | delta SawFM AUPR | delta macro-F1 | delta SawFM F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2039 | 0.840202 | 0.781040 | -0.001893 | **-0.001389** | +0.006178 | +0.016937 |
| 2040 | 0.824651 | 0.667211 | -0.004783 | **+0.007629** | +0.010404 | +0.009745 |
| 均值 | — | — | **-0.003338** | **+0.003120** | **+0.008291** | **+0.013341** |

确认集 bootstrap 95% CI：

- delta macro-AUPR：`[-0.004783, -0.001893]`；
- delta SawFM AUPR：`[-0.001389, +0.007629]`。

门槛审计：

| 条件 | 结果 | 是否通过 |
|---|---:|---|
| 平均 delta macro-AUPR >= -0.005 | -0.003338 | 是 |
| 平均 delta SawFM AUPR >= +0.005 | +0.003120 | **否** |
| macro 非劣种子 | 2/2 | 是 |
| SawFM 正收益种子 | 1/2 | **否** |

因此主门槛失败，自动决策为 `stop_static_weight_fusion`。

### 4.2 LogitFusion-0.6 上界

| Seed | delta macro-AUPR | delta SawFM AUPR | delta macro-F1 | delta Hamming | delta SawFM F1 |
|---:|---:|---:|---:|---:|---:|
| 2039 | -0.002677 | +0.024002 | -0.031781 | +0.009444 | +0.012125 |
| 2040 | -0.005096 | +0.045029 | -0.063697 | +0.022778 | +0.028915 |
| 均值 | **-0.003887** | **+0.034516** | **-0.047739** | **+0.016111** | **+0.020520** |

LogitFusion 的 SawFM 排序收益在 2/2 确认种子上复现，但 seed 2040 略超出 macro-AUPR 非劣界，而且固定阈值下 macro-F1 平均下降约 0.0477。它证明端点输出互补，却没有给出可接受的单模型决策性能。

## 5. 五种子描述性统计

以下结果合并了 3 个发现种子与 2 个确认种子，只用于估计效应方向和波动。

| 方法 | delta macro-AUPR | delta SawFM AUPR | delta macro-F1 | delta Hamming | macro 非劣 | SawFM 正收益 |
|---|---:|---:|---:|---:|---:|---:|
| WeightFusion-0.2 | -0.001457 | +0.008113 | +0.002240 | +0.002222 | 5/5 | 4/5 |
| LogitFusion-0.6 | -0.001907 | +0.037156 | -0.036488 | +0.007889 | 3/5 | 5/5 |

WeightFusion 五种子 delta SawFM AUPR 的 bootstrap 95% CI 为 `[+0.001277, +0.016936]`。这个合并区间为正，但其中包含用于选择 alpha 的发现种子，因此不能推翻独立确认失败的结论。

## 6. 失败原因分析

### 6.1 静态参数方向具有种子依赖

两个端点来自同一条训练轨迹，端点运行的 train loss 与 macro-F1 历史最大差异均为 0；因此结果差异不是重复训练漂移或数据不一致造成的。2039 的 DB-Macro/DB-Tail 最优轮次为 16/2，2040 为 20/2。固定把后期参数向第 2 轮移动 20%，在 seed 2040 上改善 SawFM 排序，在 seed 2039 上却破坏排序，说明“尾类友好方向”不是跨初始化一致的全局线性方向。

### 6.2 F1 改善不等于排序能力改善

WeightFusion 在 seed 2039 上 SawFM F1 提高 `+0.0169`，但 SawFM AUPR 下降 `-0.0014`。这表明固定阈值附近的分数位置有所变化，却没有改善全阈值范围内的排序质量。若只观察 F1，会把校准/阈值位移误判成表征提升。

### 6.3 输出互补强于参数空间可融合性

LogitFusion 在 5/5 seeds 上提高 SawFM AUPR，而 WeightFusion 只有 4/5。早期和后期模型的决策函数确实互补，但这种互补不能稳定地压缩为一条固定参数插值方向。下一阶段应直接学习“何时使用尾类残差”，而不是继续扫描静态 alpha。

### 6.4 宏观 F1 暴露决策层代价

LogitFusion 的五种子 SawFM AUPR 平均提高 `+0.0372`，同时 macro-F1 平均下降 `-0.0365`。这说明其优势主要位于排序层，统一温度和固定阈值无法把排序收益转化为整体多标签决策收益。论文中应把它定位为机制上界，而不是宣称全面性能提升。

## 7. 对长线试验设计的影响

1. **停止**在 VFT、ISRJ 上复制静态 WeightFusion-0.2；独立确认失败后继续跨类扩展会放大选择偏差。
2. **保留** Base-DB-Macro 为主要基线，LogitFusion-0.6 为双模型互补上界。
3. 下一候选应改为单模型的条件残差蒸馏：以 DB-Macro 为部署主干，以 DB-Tail 与 DB-Macro 的 logit 差作为教师残差，学习由样本特征和置信度驱动的轻量门控；训练时约束宏观蒸馏，避免统一增强尾类分数。
4. 新方法先在 SawFM 上使用新的开发 seeds 做三阶段证据链：机制 smoke、固定超参数 screen、独立确认。只有确认通过后再迁移到 VFT/ISRJ，并保持结构及超参数不按类别重调。
5. 晋级仍同时约束 macro-AUPR、目标尾类 AUPR、macro-F1、Hamming loss；不能只凭目标类 AUPR 晋级。
6. locked test 只在模型结构、超参数、种子数和统计方法全部冻结后执行一次。

建议的下一候选结构为：

```text
Base-DB-Macro backbone -> macro logits z_m
                     \-> lightweight gate g(h, confidence)
DB-Tail teacher + DB-Macro teacher -> residual target delta_z
single student output: z = z_m + g * residual_head(h)
```

该结构利用 LogitFusion 已确认的输出互补性，同时以单主干、轻量残差头替代两个完整模型和固定参数插值。它是下一轮应实现和验证的新假设，不属于本轮已验证结论。

## 8. 审计与复现

- 四个正式端点均完成，所有 run receipt 的 `locked_test_used=false`；
- 每个端点的长尾 manifest 均精确达到 2400 样本，sample count error 为 0；
- 同一 seed 的 DB-Macro 与 DB-Tail 使用相同 split SHA256 和 manifest SHA256；
- 两组端点的 `max_train_loss_difference=0`、`max_macro_f1_difference=0`；
- 每 seed 的 calibration/validation 均为 300/300，sample ID overlap 为 0；
- alpha 选择模式为 `fixed_from_prior_discovery`，本轮 `alpha_retuned=false`；
- 全量测试：`129 passed, 36 warnings`，警告均为现有 AMP API 弃用警告；
- 正式摘要 SHA256：`63AA693B90C89A5A4ACB779539BD8ACBC743F920C79408A13270D956F47F04FD`。

## 9. 产物

- 技术方案：`docs/implementation/route_a_sawfm_weight_fusion_confirmatory_plan_20260723.md`
- 批处理入口：`experiments/run_route_a_sawfm_weight_fusion_confirmatory.py`
- 固定 alpha 融合入口：`experiments/run_route_a_sawfm_checkpoint_fusion.py`
- 正式摘要：`outputs/tapc/route_a_sawfm_weight_fusion_confirmatory/route_a_sawfm_weight_fusion_confirmatory_summary.json`
- 端点运行：`outputs/tapc/route_a_sawfm_weight_fusion_confirmatory/endpoints/runs/SawFM/`
- 融合模型与清单：`outputs/tapc/route_a_sawfm_weight_fusion_confirmatory/fusion/runs/`

