# 路线 A：DFMS-IQ 冻结分类头、FPR 约束阈值与 SawFM 专家试验结果（2026-07-24）

## 1. 执行结论

本轮完成了 `3 seeds × 5 variants = 15` 个 development 评估单元，其中 H1—H4 为 12 个正式冻结头训练，J0 为三个对应 F2 联合训练检查点的原头控制。全部试验复用了上一轮对应 seed 的 split SHA256、manifest SHA256 和 F2 checkpoint；`locked_test_used=false`。

预注册的 H4 晋级门槛未通过，因此按方案没有执行末级主干解冻 U1。该停止不是工程失败，而是预注册决策规则的正常结果。

本轮最重要的结果是：**重新冻结拟合分类头本身显著释放了 F2 已有的弱证据；正样本加权 BCE 是当前最好的冻结头；本轮 batch 内弱正—难负排序和共享型 SawFM 专家没有带来独立增益。**

## 2. 固定协议

- 主干：各 seed 上一轮 `F2_DFMS_Full/best_model.pth`，完全冻结并保持 `eval()`；
- 训练清单：对应 seed 的 2400 个 label-LT 样本，确定性 80%/20% head-fit/head-validation；
- H1：随机初始化 `Linear(128,6)` + DB Loss；
- H2：随机初始化 `Linear(128,6)` + pos-weight BCE；
- H3：H2 + `0.20 ×` batch 内弱正—难负 softplus 排序，margin=0.20；
- H4：H3 + 固定权重 0.20 的 SawFM Cartesian/幅度专家；
- optimizer：AdamW，lr=1e-3，weight decay=1e-4，batch=256；
- 预算：60 epochs，patience=12，模型选择为 head-validation Macro-AUPR，同分用弱正—负 AUROC；
- 温度：独立 calibration 角色；
- 主阈值：独立 threshold-selection 角色逐标签最大化弱正召回，约束 negative FPR≤0.20；
- 诊断：early-stop development 角色，每 seed 300 样本；
- seeds：2041、2042、2043。

所有 H1—H4 的最优 epoch 都为 60，但最后五轮 validation Macro-AUPR 增量仅约 `-1.2e-6 ~ 4.6e-5`，已基本进入平台，不是明显未收敛状态。

## 3. 三随机种子正式结果

以下均为 FPR 约束阈值下的均值±标准差；Macro-AUPR 和弱正—负 AUROC 本身与阈值无关。

| 变体 | Macro-AUPR | Macro-F1 | Hamming↓ | 弱分量召回 | 弱正/负 AUROC | Negative FPR | SawFM-AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| J0 联合 DB 原头 | 0.8562±0.0037 | 0.7411 | 0.2437 | 0.3586±0.0477 | 0.5886±0.0203 | 0.1838 | 0.7329±0.0672 |
| H1 冻结重训 DB | 0.8612±0.0118 | 0.7544 | 0.2344 | 0.3641±0.0281 | 0.6396±0.0349 | 0.1803 | **0.7772±0.0723** |
| H2 pos-weight BCE | **0.8670±0.0057** | **0.7554** | 0.2356 | **0.3774±0.0402** | **0.6524±0.0349** | 0.1875 | 0.7709±0.0549 |
| H3 +弱正难负排序 | 0.8643±0.0019 | 0.7502 | 0.2378 | 0.3675±0.0312 | 0.6513±0.0323 | 0.1822 | 0.7607±0.0522 |
| H4 +SawFM专家 | 0.8654±0.0026 | 0.7513 | 0.2367 | 0.3705±0.0443 | 0.6515±0.0290 | 0.1825 | 0.7584±0.0508 |

### 3.1 H1 相对 J0：冻结重拟合头有效

配对均值变化：

- Macro-AUPR：`+0.00499`；
- Hamming：`-0.00926`；
- 弱正—负 AUROC：`+0.05104`，3/3 seeds 为正；
- 弱正—负 AUPR：`+0.03822`；
- SawFM-AUPR：`+0.04431`；
- negative FPR：`-0.00348`。

弱分量召回仅增加 `+0.00543`，说明“F2 有弱证据但原联合头没有充分利用”的判断成立；然而只重新拟合 DB 头还不足以把排序改善完全转换为受控召回。

### 3.2 H2 相对 H1：pos-weight BCE 是有效的下一头部基线

- Macro-AUPR：`+0.00582`；
- 弱分量召回：`+0.01335`，2/3 seeds 为正；
- 弱正—负 AUROC：`+0.01274`，3/3 seeds 为正；
- negative FPR：`+0.00722`；
- Hamming：`+0.00111`，变化很小；
- SawFM 弱正—负 AUROC：`+0.02983`，但 SawFM-AUPR `-0.00632`。

逐标签上，H2 的弱正—负 AUROC 相对 H1 在 SawFM、MFT、VFT、ISRJ 分别提高约 0.0298、0.0181、0.0191、0.0108；FMB 和 SinFM 略降。因此 H2 改善了最需要弱证据支持的多类，但不是所有类别统一改善。

H2 相对旧 J0 的均值变化约为：Macro-AUPR `+0.01082`、弱正—负 AUROC `+0.06378`、弱分量召回 `+0.01878`。当前应把 H2 作为冻结头阶段的主要候选，而不是 H4。

### 3.3 H3 相对 H2：当前排序损失失败

- Macro-AUPR：`-0.00271`；
- 弱分量召回：`-0.00994`；
- 弱正—负 AUROC：`-0.00111`；
- SawFM-AUPR：`-0.01022`；
- SawFM 弱正—负 AUROC：`-0.02497`；
- negative FPR：`-0.00531`。

排序项略微降低误报，但没有改善弱正排序，尤其伤害 SawFM。主要原因不是“排序思想必然无效”，而是本轮实现采用 batch 内 top-k 难负：

1. SawFM 弱正数少，batch 内正负配对方差较大；
2. top-k 负样本可能包含结构上接近但并非同一可见性难度的样本，产生不稳定梯度；
3. 所有标签共享同一 λ 和 margin，忽略 FMB/SinFM 与 MFT/VFT/SawFM 的可见性差异；
4. head-validation 仍以 Macro-AUPR 选点，排序项的局部目标没有获得独立选择保障。

因此不能据此宣称弱正排序机制有效，也不应把 H3 接入主干微调。

### 3.4 H4 相对 H3：当前 SawFM 专家贡献接近零

- Macro-AUPR：`+0.00103`；
- Hamming：`-0.00111`；
- 弱分量召回：`+0.00305`；
- 弱正—负 AUROC：`+0.00022`；
- SawFM-AUPR：`-0.00228`；
- SawFM 弱正—负 AUROC：`+0.00412`。

该专家只是从同一 F2 深层复数特征中移除相位残差，仍共享 temporal block、statistics pool 和 feature projection，因此它不是一个真正独立的 SawFM 证据提取器。固定 0.20 融合避免了过度放大，但也只能对已有的高度相关表示做小幅修正。结果不支持把当前专家作为创新模块。

## 4. FPR 约束阈值结果

在 `15 variants/seeds × 6 labels = 90` 个 threshold-selection 单元中，negative FPR≤0.20 的约束违规数为 0，说明选择器工程实现正确。

相对逐标签 F1 阈值，FPR 约束阈值产生了预期的“召回换误报”收缩。以 H2 为例：

- 弱分量召回：`-0.1021`；
- negative FPR：`-0.0813`；
- Macro-F1：`-0.0100`；
- Hamming：`-0.0111`，即总体逐标签错误率反而下降。

因此该阈值的意义不是制造更高的绝对召回，而是在统一误报预算下比较不同分类头。H2 在受控 FPR 下仍取得本轮最高弱分量召回，因而其改进不是简单降阈值造成的。

threshold-selection 到 early-stop 存在有限漂移：90 个标签单元中有 5 个 early FPR 超过 0.25，全部发生在 seed=2043 的 FMB 或 SinFM；对应 threshold-role FPR 均不超过 0.20。这说明下一轮若继续优化阈值，应使用有限样本置信上界（例如 Wilson upper bound）而不是继续提高 FPR 上限。

## 5. 预注册晋级门槛

H4 相对 H1：

- Macro-AUPR `+0.00414`：通过非劣门槛；
- Hamming `+0.00222`：通过；
- negative FPR `+0.00222`，H4 绝对值 0.1825：通过；
- 弱正—负 AUROC `+0.01185`：低于要求的 +0.015；
- 弱分量召回 `+0.00646`：低于要求的 +0.050，且仅 1/3 seeds 为正；
- SawFM 弱正—负 AUROC `+0.00898`：低于要求的 +0.020；
- SawFM-AUPR `-0.01881`：未通过非劣要求。

最终门槛：**未通过**。正式决策：`stop_before_unfreeze`。因此没有执行 U1，也没有以小学习率修改 F2 的 stage4 或 temporal block。

## 6. 当前有效结论与下一决策

### 已验证

1. F2 表示中确实保留了可被重新分类的弱证据；旧联合 DB 头是明显瓶颈。
2. 冻结 F2 后，pos-weight BCE 线性头是本轮最好的综合候选。
3. FPR 约束阈值能在 threshold-selection 角色严格控制误报，并避免用无约束降阈值夸大弱召回。
4. 当前 batch 内统一弱排序损失无效。
5. 当前共享型 Cartesian/幅度 SawFM 专家无独立有效性证据。

### 下一轮建议

1. 暂不解冻主干，先把 H2 扩展到五随机种子确认；主要目标是确认相对 J0/H1 的弱 AUROC、Macro-AUPR 和 FPR 稳定性。
2. 阈值改为基于 negative FPR 置信上界的保守选择，专门处理 seed=2043 的 FMB/SinFM 漂移。
3. 若再次研究排序损失，应改为离线跨 batch 难负记忆库，并只对 MFT/VFT/ISRJ 或通过训练集可辨识性筛选后的标签启用；不可继续统一 λ/margin。
4. 若再次研究 SawFM 专家，应从更早的复数 stage3/stage4 分叉，使用独立的幅度包络、瞬时频率或时频统计池化；当前“只去除晚期 phase 残差”的专家应停止。
5. 只有新的冻结头在五种子下通过晋级门槛，才重新考虑末级小学习率解冻。

## 7. 实验产物

- 方案：`docs/implementation/route_a_dfms_iq_frozen_head_and_saw_expert_plan_20260724.md`
- 执行器：`experiments/run_route_a_dfms_iq_frozen_heads.py`
- 汇总：`outputs/tapc/route_a_dfms_iq_frozen_heads/route_a_dfms_iq_frozen_heads_summary.json`
- 单次产物：`outputs/tapc/route_a_dfms_iq_frozen_heads/seed_<seed>/<variant>/`
- GPU smoke：`outputs/tapc/route_a_dfms_iq_frozen_heads_smoke/`

这些结果仍是 development 证据，不是锁定测试集结果，也不能直接作为最终论文性能数字。
