# Route A H3：VFT 可见性三层身份配对可辨识性验证

状态：预注册，尚未生成正式源、尚未训练。  
前序依据：[H2 多脉冲结果](../results/route_a_h2_multipulse_sixlabel_results_20260803.md) 已排除“当前真实时间戳专家的加性融合自然优于同容量置换专家”。H2 的 development 源没有 strong-VFT-positive 层，因此不能区分“时序特征不可辨识”与“弱/强/负三层训练合同缺失”。

## 1. 单一可证伪假设

在相同 receiver identity、相同背景复合干扰和相同噪声的三臂反事实组中，真实时间戳 VFT 统计应同时满足：

1. 弱 VFT（8 dB）相对 composition-matched VFT-absent hard negative 有稳定排序优势；
2. 强 VFT（20 dB）相对 hard negative 的排序不弱于弱 VFT；
3. 真实时间戳统计的优势大于完全同维度、同参数、时间戳置换的统计。

该阶段不训练 VTSE-VBF、不调融合系数、不访问任何 external、historical locked test、test-v2 或 test-v3。它只回答“适合三层可见性建模的 VFT 物理证据是否存在”。

## 2. 数据合同

每个 `pair_seed` 生成三个**共享身份**的视图：

| arm | `vft_visibility_arm` | VFT 标签 | VFT JNR | 作用 |
|---|---:|---:|---:|---|
| absent | 0 | 0 | NaN | composition-matched hard negative |
| weak | 1 | 1 | 8 dB | 标签内弱正 |
| strong | 2 | 1 | 20 dB | 标签内强正 |

一个三臂组的五个非 VFT 成分、非 VFT 标签、非 VFT 功率、VFT Doppler/arrival 参数和复高斯 receiver noise 全部相同；唯一因果改动是 VFT 成分的存在及其相对功率。背景五个标签仍按固定生成分布抽取，且至少一个背景分量存在。九挡或同身份多视图永远按 `pair_seed` 分组，绝不跨训练/校准/阈值/评估角色拆开。

计划角色（均为新 receiver identity block）：

| 角色 | 样本 / 三臂组 | base identity | 用途 |
|---|---:|---:|---|
| `train-3strata-v1` | 6,048 / 2,016 | 3,200,000,000 | 仅拟合 |
| `development-3strata-v1` | 3,024 / 1,008 | 3,300,000,000 | 源审计、固定机制评价、后续五种子 development |

H3 这一可辨识性阶段**不生成 test/external role**，以避免在尚未证明物理合同前消耗独立证据。正式源必须与 H0q/H1a/H1b、H2 三个角色、历史训练身份和 test-v3 完全无 receiver identity 重叠。

## 3. 先验实现与审计

生成器必须写入：I/Q、6-label labels、`component_jnr_db`、`pair_seed`、`receiver_noise_seed`、`vft_visibility_arm`、SHA-256 generation receipt。源审计必须验证：

1. 每个组恰为 3 行，receiver identity 共享；
2. arm 为 `{0,1,2}`，标签为 `{0,1,1}`，VFT 功率为 `{NaN,8,20}`；
3. 非 VFT labels/powers 在三行逐元素相同；
4. 信号 shape、数值有限、历史/H0q/H1a/H1b/H2 identity overlap 均为零；
5. receipt 中所有测试使用字段均为 false。

先运行一个单独命名的 30-view / 10-group smoke，仅验证代码和审计链。Smoke 不得用于任何门槛、模型选择或方法有效性结论。只有 smoke 和正式 source audit 都通过，才启动后续固定五种子统计。

在任何可训练模型之前，development-3strata 的无训练物理统计使用固定的八脉冲归一化相干全栅格最大值，比较真实时间戳与 H2 相同的固定置换 `[0,5,2,7,4,1,6,3]`。其**预先固定**门槛为：弱正-vs-absent 真时间戳 AUROC `>=0.58`；真减置换 AUROC `>=+0.040` 且三臂 identity-paired bootstrap 95% CI 下界 `>0`；强正-vs-absent 真时间戳 AUROC 不得比弱正低 `>0.010`；同 identity 的弱正得分高于 absent 得分比例 `>=0.56`。任一项失败即停止 H3，不实现 T0/T1/T2。

## 4. 后续模型验证（尚未授权执行）

只有 H3 源合同通过后，才实现下列固定三臂对照：

- `T0`：共享多脉冲 baseline；
- `T1`：T0 + 真实时间戳 VFT 统计投影；
- `T2`：T0 + 时间戳置换、同 25 维输入、同 `25→16→1` 投影的容量对照。

所有臂使用同一输入、optimizer、24-epoch 上限、BCE 和五个新训练种子。T1/T2 均只允许改写 VFT logit。主指标为 weak-vs-absent AUROC/AUPR；强-vs-absent AUROC 是强正安全指标；其他五个标签必须逐 logit 不变。

训练合同固定为：seeds `20261003, 20261011, 20261019, 20261027, 20261104`；AdamW（LR `3e-4`，weight decay `1e-4`），batch `16`，global gradient clip `5`，pulse-backbone width `32`，VFT projection `25→16→1`，BCEWithLogitsLoss，无数据增强。每臂最多 24 epoch，development raw macro-AUPR 最高的 epoch 仅用于保存 development checkpoint；所有主张明确为 development 机制证据。无校准器、无阈值、无 F1 选择。每次运行保存配置、源 SHA-256、history、checkpoint、raw predictions、逐标签指标和 receipt。

晋级门槛预注册为：T1 相对 T2 的弱 AUROC 平均增益至少 `+0.030`、5/5 为正、paired bootstrap CI 下界大于零；相对 T2 的弱 AUPR 平均增益至少 `+0.025` 且 CI 下界大于零；相对 T0 的强 AUROC 平均差不低于 `−0.010`、每个种子不低于 `−0.020`；T1/T2 参数量严格相等；且源层级的真实时间顺序统计相对置换存在同方向效应。任一条件失败即停止 H3，不调时间戳排列、ROI、专家宽度、JNR、轮数、学习率或损失。

## 5. 证据边界

H3 成功只能证明：三层反事实物理合同下存在超越置换容量对照的 VFT 时间证据。它仍不证明完整 VTSE-VBF、MFT/ISRJ 改进、主流基线优越性、外部泛化或 test-v3 性能；这些均须在后续新的统一候选协议中完成。
