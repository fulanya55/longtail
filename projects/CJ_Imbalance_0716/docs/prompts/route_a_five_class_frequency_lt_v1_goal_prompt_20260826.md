# Route A 五类频次长尾实验 Goal Prompt

你是本项目的实验执行代理。当前任务只研究“去除 SawFM 后的五类复合干扰识别”与“标签频次长尾”问题，不研究功率长尾、弱分量可见性、半监督学习或开集识别。

## 研究对象与边界

- 工程根目录：`E:\LJQ_code\CJ_Imbalance_0716`。
- 数据根目录：`E:\LJQ_code\干扰数据\复合干扰数据0716`。
- Python 环境固定为 `PY10`：`D:\anaconda3\envs\PY10\python.exe`。
- 原始类别顺序为 `FMB, SawFM, SinFM, MFT, VFT, ISRJ`。
- 本阶段只保留五类 `FMB, SinFM, MFT, VFT, ISRJ`；训练集与测试集均删除 SawFM 阳性样本，而不是把 SawFM 阳性样本错误地标成五类负样本。
- 所有样本使用原始 0716 闭集的 `complex_rms` 归一化；不使用九档功率扫描，不构造 VL/VH 配对，不按功率重新采样。
- 当前测试集只能作为独立结果集使用，不能用于模型、损失、阈值或超参数选择；不得访问已经封存的 locked-test。

## 固定骨干网络

使用当前 Route A 已验证的 B0 `DelayedFusionMultiScaleIQClassifier`（`dfms_iq`），输入 `[B,2,3000]`：复数实部/虚部分离的四级卷积、多尺度 stage3/stage4（kernel 3/7/15，dilation 1/2/4）、延迟 Cartesian/幅度/相位融合、128 通道多尺度深度可分离时序残差块、mean/std 全局池化、256→128 特征投影和五类线性分类头。宽度 16、特征维度 128、dropout 0.2；本阶段不扩宽主干、不加入 SawFM 专家、不加入功率条件输入。

## 频次长尾协议

1. 对训练源先过滤 SawFM 阳性样本。
2. 在五类标签上建立只含标签频次的 `LabelIF4` 派生 manifest（目标头尾正样本数比为 4:1）。原始 0716 组合表是完全对称的，直接强行 IF20 会导致整数组合约束不可行；首轮采用可审计、可达的 IF4，后续再用独立生成源验证更高 IF。记录实际达到的计数、组合计数、源文件 SHA256、样本索引 SHA256。
3. 另外保留一个五类频次平衡训练控制（相同过滤、相同输入与训练预算）。
4. 每个 seed 使用相同的训练/early 划分规则；test-v1 只在所有选择完成后评估。

## 必做对照

- `B0-Focal-Balanced`：五类平衡训练控制。
- `B0-Focal-LabelIF4`：不改损失的频次长尾基线。
- `B0-DB-LabelIF4`：Distribution-Balanced Loss 的五类频次长尾候选。
- 如计算预算允许，增加 `B0-DB-Balanced`，用于区分“骨干/优化收益”和“频次长尾收益”。

所有对照必须保持输入、主干、batch、epoch、优化器、学习率、归一化和 early 选择指标一致；只能改变训练样本 manifest 与损失。

## 训练参数

- seeds：`271001, 271002, 271003`；如某 seed 失败，保留失败 receipt，不替换 seed。
- AdamW；backbone lr `5e-5`，classifier lr `5e-4`，weight decay `1e-4`，gradient clip `5.0`。
- batch size `64`；最大 epoch `14`；early patience `5`；early/evaluation batch `256`；num_workers `0`。
- Focal：`gamma_pos=0, gamma_neg=3`。
- DB：使用训练 manifest 的五类实际正样本计数，保持代码默认 DB 参数，不手工按测试结果调参。
- 模型选择指标仅用 early macro-AUPR；阈值不参与模型选择。

## 结果与停止规则

至少保存：数据过滤审计、长尾 manifest、每 seed 每 arm 的 checkpoint、训练 history、early/test logits、每类 AUPR/AUROC/F1、macro-AUPR、macro-F1、标签计数、参数量、脚本与配置 SHA256、terminal receipt、失败原因。

首轮只回答三个问题：

1. 去除 SawFM 后，B0 是否仍能稳定学习五类频次长尾？
2. DB 相比 Focal 是否在尾类 AUPR 和 macro-AUPR 上产生跨 seed 一致增益？
3. 增益是否来自频次长尾损失，而不是测试集调参或功率轴泄漏？

建议的候选准入门槛：LabelIF4 macro-AUPR 相对 Focal-LabelIF4 平均增益 ≥0.010，至少 2/3 seed 同向，尾部两类平均 AUPR 增益 ≥0.020，且头部类下降不超过 0.010；否则标记为负结果，进入损失/采样诊断，不得声称五类长尾问题已解决。

## 执行顺序

先运行数据审计和 dry-run；通过后生成 manifest；然后执行首轮三 seed 四臂训练；最后生成中文报告。若遇到错误，先修复活动脚本并重跑最小复现，不能修改历史报告、封存测试或已完成 receipt。

最终报告必须明确区分 `[F]` 已验证、`[A]` 当前开发证据、`[N]` 失败/停止、`[T]` 尚未验证的计划，并给出下一轮仅针对频次长尾的改进建议（重采样、logit adjustment、cRT、class-balanced loss、label-aware sampler 的优先级与消融顺序）。
