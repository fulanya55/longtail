# P1g：ISRJ 逐功率置信缓冲校准确认试验

日期：2026-07-31  
证据等级：全新身份隔离 development-confirmation；非 confirm/locked test  
结论：预冻结的“风险校准创新点”通过；原 P1f“置信下界路由创新点”仍未通过。

## 1. 最近三轮试验及其递进关系

| 轮次 | 核心问题 | 主要结果 | 成果 | 失败原因/边界 |
|---|---|---:|---|---|
| P1d | 任务路由能否在新独立数据上兼顾召回和安全 | ISRJ Recall 增益约 `+0.08264`；20–25 dB 最坏 FPR `0.11129` | 证明新数据上存在稳定召回收益 | 超过 `FPR<=0.10` 部署门槛；不能进入 P2，也不能在该测试集上继续调参 |
| P1e | 失败来自专家上限不足还是路由不安全 | ISRJ oracle Recall 上限 `+0.15417`；物理联合路由 `+0.08854`；最大 FPR 增量 `+0.02258` | 排除了“ISRJ 专家没有增益空间”这一解释 | MFT/VFT 专家上限不足；ISRJ 的主要矛盾转为安全校准，而非继续增加路由特征 |
| P1f | 置信下界路由能否优于均值路由 | lower route Recall `+0.07435`，最坏 FPR `0.04274`，AUPR `+0.00212`，漏检惩罚 NMAE `-0.00270` | 3/3 seeds 同方向，实用指标安全 | Recall 未达到 `+0.10`；且未相对 mean route 获得一致安全优势，增益主要来自校准阈值而非“置信下界”机制 |

因此，P1g 不再声称置信下界路由本身是创新，而把可证伪命题收窄为：

> 在模型、路由分数和最终概率完全相同的条件下，逐功率有限样本置信缓冲校准能否相对普通 pooled-FPR 校准降低独立测试的最坏功率 FPR，并保留主要召回收益？

## 2. P1g 冻结方案

### 2.1 数据与身份隔离

- 新生成根目录：`generated_data/p1g_isrj_risk_confirmation_seed400000000`
- 重分块目录：`generated_data/p1g_isrj_risk_confirmation_seed400000000_rechunked`
- 功率：20 dB、25 dB。
- 每个 split、每个功率：63 个非空六类组合 × 40 个身份 = 2520 个样本。
- fresh-train 与 fresh-test 身份数均为 2520。
- train/test 身份重叠：0；fresh-test 与原始 0716 身份重叠：0。
- fresh-train 内按每种组合 `20/10/10` 划分 fit/selection/calibration；三个角色互斥。

### 2.2 固定模型与路由

- 分类/参数主干：固定的 `MLCNN IQ-Grouped task-routed` 参数目标 checkpoint，seeds 2041/2042/2043。
- 物理专家：固定 conditional U-Net 分离器与 ISRJ 周期性特征。
- 路由输入：主干概率 + ISRJ periodicity。
- 集成：15 个按身份分组 bootstrap 的 risk/benefit 模型。
- 路由分数：10% 分位置信下界；混合系数 `alpha=0.75`。
- 路由选择和所有阈值只使用 fresh-train 内部角色。

### 2.3 唯一实验变量

同一 lower-route 最终概率分别使用两种阈值规则：

1. **逐功率置信缓冲（提出方法）**：在 calibration 上要求每个功率同时满足观测 FPR `<=0.06`、Clopper–Pearson 95% 上界 `<=0.08`，再选择召回最高的阈值。
2. **普通 pooled-FPR（强对照）**：把两个功率的 calibration 负样本合并，只约束总体 FPR `<=0.10`。

冻结哈希：`a8855e5bc4560ab86f83171a3ef6fd7266ace70fe49d24ffc87e5707de7ec928`。冻结文件写入时 `test_used_at_freeze=false`；之后 fresh-test 只使用一次，无 test fit/selection。

### 2.4 预冻结创新验证门槛

- 缓冲校准平均 Recall 增益 `>=0.05`；
- 3/3 seeds Recall 增益均为正；
- 独立测试最大功率观测 FPR `<=0.10`；
- macro-AUPR 相对基线不劣于 `-0.005`；
- 参数漏检惩罚 NMAE 不恶化；
- 至少保留 pooled 校准 70% 的平均 Recall 增益；
- 安全优势：缓冲校准最坏 FPR 至少比 pooled 低 `0.02`，或 pooled 违反 0.10 而缓冲通过。

## 3. 独立测试结果

### 3.1 聚合结果

| 指标 | 逐功率置信缓冲 | pooled-FPR 对照 | 结论 |
|---|---:|---:|---|
| 平均 ISRJ Recall 增益 | `+0.07656` | `+0.07630` | 缓冲方法未牺牲平均收益 |
| 三个 seed 的 Recall 增益 | `+0.06406 / +0.04883 / +0.11680` | `+0.09180 / +0.07695 / +0.06016` | 缓冲方法 3/3 为正 |
| 最坏功率观测 FPR | `0.09274` | `0.11613` | 绝对下降 `0.02339`，并从失败变为通过 0.10 门槛 |
| Recall 收益保留率 | `100.34%` | 基准 | 高于冻结要求 70% |
| macro-AUPR 增量 | `+0.00192` | 同一概率，数值相同 | 非劣 |
| 参数漏检惩罚 NMAE 增量 | `-0.00262` | `-0.00252`（三 seed 均值） | 未恶化，负值表示改善 |

全部七项冻结门槛通过，`risk_calibration_confirmation.innovation_validated=true`。

### 3.2 逐 seed、逐功率风险

| Seed | 校准方式 | 20 dB FPR | 25 dB FPR | 最大 FPR | Recall 增益 |
|---:|---|---:|---:|---:|---:|
| 2041 | 逐功率缓冲 | 0.04274 | 0.02661 | 0.04274 | +0.06406 |
| 2041 | pooled | 0.10323 | 0.09355 | 0.10323 | +0.09180 |
| 2042 | 逐功率缓冲 | 0.04113 | 0.03629 | 0.04113 | +0.04883 |
| 2042 | pooled | 0.07984 | 0.07984 | 0.07984 | +0.07695 |
| 2043 | 逐功率缓冲 | 0.08629 | 0.09274 | 0.09274 | +0.11680 |
| 2043 | pooled | 0.11048 | 0.11613 | 0.11613 | +0.06016 |

逐 seed 观察可见，缓冲校准在三个 seed 上都降低了各自的最坏 FPR；2041 和 2043 把 pooled 的超限风险拉回 0.10 以内。2043 中缓冲方法还同时提高了相对基线的 Recall 增益，说明高阈值并不必然损失路由净收益，原因是两种校准各自对应的基线阈值也按相同规则公平重校准。

## 4. 成功原因与失败项解释

### 4.1 为什么 P1g 成功

1. **研究问题被正确收窄。** P1e/P1f 已显示 ISRJ 具有专家上限和稳定召回空间，但“lower 比 mean 更安全”没有得到支持。P1g 只验证真正未解决的校准机制。
2. **对照消除了模型混杂。** 两组共享模型、专家、路由接受样本和最终概率，区别只有阈值校准规则，因此最坏 FPR 的变化可归因于校准规则。
3. **逐功率约束针对 pooled 校准的结构性盲点。** pooled 允许一个功率档集中承担错误；缓冲规则要求每个档位单独通过，并使用有限样本置信上界预留外推余量。
4. **身份隔离和一次性测试限制了选择偏差。** 新 seed 范围与 0716/P1d/P1f 分离，测试在完整冻结后才读取。

### 4.2 仍然失败或不能声称的部分

- P1f 的“置信下界路由优于均值路由”仍失败：P1g lower route 平均 Recall `+0.07656`，仍未达到旧门槛 `+0.10`，且最坏 FPR 与 mean route 都是 `0.09274`。
- 测试集上最坏 Clopper–Pearson 95% 上界为 `0.10742`，高于 0.10；本轮通过的是预冻结的**观测 FPR**门槛，不应表述为“95% 置信下部署风险必然低于 0.10”。
- 本轮只覆盖 20/25 dB 和当前六类组合生成分布；尚未证明对其他功率、真实采集域或不同干扰参数先验泛化。
- 本轮不是 locked test，不能替代最终一次性锁定测试；也不能再用本 P1g test 调阈值或选择结构。

## 5. 当前可支持的创新表述

可支持的窄创新点：

> 面向复合干扰任务路由的逐功率有限样本置信缓冲校准，在全新身份隔离的 20/25 dB 开发确认集上，相对普通 pooled-FPR 校准将跨功率最坏观测 FPR 从 0.11613 降至 0.09274，同时保留 100.34% 的平均 Recall 增益，三个随机种子均保持正向收益，且未损害 macro-AUPR 与参数漏检惩罚指标。

不能支持的表述包括：“置信下界路由本身显著优于均值路由”“所有功率/真实域均满足 FPR<0.10”“已经通过 locked test”。

## 6. 下一步长线验证计划

1. **冻结 P1g 全部规则。** 不再使用 P1g test 做任何阈值、特征、功率缓冲或模型选择。
2. **扩大但仍独立的开发确认。** 用新身份覆盖 10–50 dB 九档功率，预注册主要指标为跨功率 worst-FPR、Recall 收益和其 paired-bootstrap 区间；保持逐功率规则不变。
3. **强化有限样本风险证据。** 增加每功率校准负样本量，或预注册更严格的 calibration CP 上界，使独立测试的 CP95 上界也以 `<=0.10` 为目标。
4. **验证参数分布外泛化。** 对 ISRJ 转发次数、切片宽度/周期、弱分量功率分别做 identity-safe 留区间验证，检查安全收益是否依赖当前先验。
5. **计算代价与部署。** 报告分离器+主干双骨干的参数量、FLOPs、显存和端到端延迟，并与仅概率校准基线对照。
6. **最终 locked test。** 只有上述方法、规则、统计脚本和模型选择完全冻结并在新的多功率开发确认中通过后，才申请一次性 locked test。

## 7. 复现实物

- 方法与训练脚本：`cj0724/p1f_isrj_confidence_bound_router.py`
- P1g 数据生成：`matlab/generate_p1g_isrj_risk_confirmation_data.m`
- 重分块脚本：`cj0724/rechunk_independent_power_data.py`
- 单元测试：`tests/test_p1f_isrj_confidence_bound_router.py`
- 正式输出：`outputs/p1g_isrj_risk_calibration_confirmation_20260731`
- 核心摘要：`outputs/p1g_isrj_risk_calibration_confirmation_20260731/summary.json`
- 逐样本预测：各 `seed_*/test_per_sample.npz`
- 数据生成回执：`generated_data/p1g_isrj_risk_confirmation_seed400000000/generation_receipt.mat`
- 重分块回执：`generated_data/p1g_isrj_risk_confirmation_seed400000000_rechunked/rechunk_receipt.json`

本报告只把 P1g 作为 development-confirmation 证据；`confirm_used=false`、`locked_test_used=false`。
