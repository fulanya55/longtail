# Legacy 多任务改造与学习率小范围调优结果（2026-07-25）

## 结论

- 8 个 Legacy 兼容骨干均已接入统一多任务接口，输出形状为分类 `[B, 6]` 与参数 `[B, 6, 2]`。
- 分类方面，`legacy_mlcnn` 是本轮最优骨干；3 seeds confirm 的 Macro-AUPR 为 `0.9299±0.0030`，固定阈值 Macro-F1 为 `0.8314±0.0055`。
- 参数方面，所有候选都未相对训练均值常数基线达到 10% NMAE 降幅。当前最优是 `iq_grouped`，降幅约 `7.28%`；Legacy 最优 `legacy_mlcnn` 约 `5.06%`。因此 Legacy 参数输出已实现并可训练，但尚不能声称参数感知已通过重新接回 C2 的门槛。
- ConvNeXt 的最佳小范围配置为 `2 epoch warmup + Cosine, lr=3e-4, wd=1e-4`；`wd=1e-5` 与 `1e-4` 几乎没有可分辨差异。
- EfficientNet-B0/B1 在搜索配置及 3/3 seeds 上都接近随机排序，不能根据自适应阈值后的表面 Macro-F1 判定成功。
- 自适应阈值提高 Macro-F1，但多数模型的 subset accuracy 下降。需要按最终目标选择：类别级 F1 优先时使用校准阈值，严格复合标签全匹配优先时保留固定阈值或另行优化组合级目标。

## 改造内容

1. 沿用每个 Legacy 模型已有的原始特征抽取和 I/Q 输入适配器。
2. 统一接入 256 维特征适配器、六标签分类头和六个二参数回归分支。
3. 参数损失仅在 `label=1 AND parameter_mask=1` 的有效位置计算 Smooth-L1；无效 NaN 不参与损失。
4. 多标签指标支持标量阈值或六类别阈值向量。
5. 新训练入口支持：
   - 2 epoch 线性 warmup；
   - Cosine 衰减至基础学习率的 5%；
   - warmup 后 ReduceLROnPlateau，factor 0.5、patience 2；
   - 独立早停、阈值校准、调参选择和 confirm 数据角色；
   - 校准集逐类别搜索 0.05 至 0.95、步长 0.01 的最大 F1 阈值。

## 数据与选择协议

- 数据：0716 平衡根训练集。
- 每个 63 种复合标签组合在本次小筛中分别抽取：训练 20、早停 5、校准 5、调参 5、confirm 5。
- 总样本角色：训练 1260、早停 315、校准 315、调参 315、confirm 315，五者完全互斥。
- checkpoint 只由早停角色的联合损失选择。
- 自适应阈值只在校准角色拟合；调参角色用于选择超参数；confirm 仅作根训练集内独立确认。
- 未使用长尾采样，未使用 locked test。

## EfficientNet 与 ConvNeXt 小范围搜索

单种子 2041，10 epochs，比较以下五种配置：

- Cosine：`lr={1e-4, 3e-4} × wd={1e-5, 1e-4}`；
- ReduceLROnPlateau：`lr=1e-4, wd=1e-4`。

| 模型与选定配置 | Confirm AUPR | 自适应 F1 | 固定 F1 | 自适应 subset | 参数 NMAE |
|---|---:|---:|---:|---:|---:|
| ConvNeXt, Cosine 3e-4 / 1e-4 | 0.6674 | 0.7059 | 0.6658 | 0.0444 | 0.7944 |
| ConvNeXt, Plateau 1e-4 / 1e-4 | 0.6375 | 0.6948 | 0.6557 | 0.0286 | 0.8023 |
| ConvNeXt, Cosine 1e-4 / 1e-4 | 0.6206 | 0.6927 | 0.6345 | 0.0381 | 0.8120 |
| EfficientNet-B0, Cosine 1e-4 / 1e-4 | 0.5044 | 0.6737 | 0.4491 | 0.0159 | 0.8376 |
| EfficientNet-B1, Cosine 1e-4 / 1e-4 | 0.5085 | 0.6737 | 0.3368 | 0.0159 | 0.8493 |

ConvNeXt 的 3-seed confirm 为：AUPR `0.6751±0.0325`、自适应 F1 `0.7039±0.0055`、固定 F1 `0.6547±0.0089`、参数 NMAE `0.7933±0.0036`。

EfficientNet 的 3-seed AUPR 分别只有 B0 `0.5079±0.0075`、B1 `0.5129±0.0061`，且 subset 均为 `0.0159`。自适应 F1 恒为 `0.6737` 是阈值把无区分度输出推向高阳性覆盖后的退化解，不代表分类排序能力提升。

## Legacy 单种子筛选

统一采用 `2 epoch warmup + Cosine, lr=1e-4, wd=1e-4`。

| Legacy 骨干 | Confirm AUPR | 自适应 F1 | 固定 F1 | 固定 subset | 参数 NMAE |
|---|---:|---:|---:|---:|---:|
| MLCNN | 0.9272 | 0.8350 | 0.8243 | 0.3175 | 0.7936 |
| ResNet34 | 0.8889 | 0.8101 | 0.7968 | 0.2730 | 0.8032 |
| EfficientNet-v2 | 0.8471 | 0.7565 | 0.7478 | 0.1619 | 0.7996 |
| EfficientNet-v1 | 0.8422 | 0.7511 | 0.7582 | 0.1810 | 0.7974 |
| EfficientNet-v3 | 0.8344 | 0.7483 | 0.7402 | 0.1619 | 0.7998 |
| MobileNetV2 | 0.8086 | 0.7388 | 0.7255 | 0.1270 | 0.8079 |
| EfficientNet-2D-IQM | 0.7845 | 0.7148 | 0.6854 | 0.1048 | 0.7946 |
| ShuffleNetV2 | 0.7651 | 0.7373 | 0.7178 | 0.0825 | 0.8062 |

## 3-seed 公平比较

所有模型使用相同数据角色和 `Cosine lr=1e-4, wd=1e-4`；ConvNeXt 使用其搜索选出的 `lr=3e-4`。

| 模型 | AUPR | 自适应 Macro-F1 | 固定 Macro-F1 | 自适应 subset | 固定 subset | 参数 NMAE |
|---|---:|---:|---:|---:|---:|---:|
| Legacy MLCNN | 0.9299±0.0030 | 0.8384±0.0026 | 0.8314±0.0055 | 0.2847±0.0324 | 0.3249±0.0105 | 0.7982±0.0044 |
| Legacy ResNet34 | 0.8821±0.0048 | 0.7971±0.0094 | 0.7823±0.0114 | 0.2159±0.0250 | 0.2635±0.0206 | 0.8012±0.0071 |
| Complex1D | 0.8434±0.0077 | 0.7456±0.0012 | 0.7299±0.0053 | 0.1005±0.0176 | 0.1619±0.0119 | 0.7840±0.0058 |
| IQ-Grouped | 0.8300±0.0044 | 0.7344±0.0026 | 0.7218±0.0017 | 0.0847±0.0075 | 0.1376±0.0191 | 0.7796±0.0032 |
| ConvNeXt | 0.6751±0.0325 | 0.7039±0.0055 | 0.6547±0.0089 | 0.0423±0.0030 | 0.0487±0.0054 | 0.7933±0.0036 |
| EfficientNet-B1 | 0.5129±0.0061 | 0.6737±0.0000 | 0.2994±0.0529 | 0.0159±0.0000 | 0.0159±0.0000 | 0.8444±0.0035 |
| EfficientNet-B0 | 0.5079±0.0075 | 0.6737±0.0000 | 0.3368±0.0917 | 0.0159±0.0000 | 0.0159±0.0000 | 0.8400±0.0017 |

训练均值常数预测在 confirm 上的参数 Macro-NMAE 为 `0.84077`。相对降幅：IQ-Grouped `7.28%`、Complex1D `6.75%`、ConvNeXt `5.65%`、Legacy MLCNN `5.06%`、Legacy ResNet34 `4.71%`。均未达到 10% 参数晋级线。

## 阈值比较与使用建议

- MLCNN 的自适应阈值使 Macro-F1 平均提高约 `0.0070`，但 subset 从 `0.3249` 降到 `0.2847`。
- ResNet34 的 Macro-F1 提高约 `0.0148`，subset 从 `0.2635` 降到 `0.2159`。
- ConvNeXt 的 Macro-F1 提高约 `0.0492`，但其 AUPR 仍明显低于 MLCNN、ResNet34 和两个标准轻量骨干。
- EfficientNet 的自适应 F1 不可采信为成功证据，应以 AUPR、固定阈值表现和 subset 的共同失败为准。

推荐默认保留并同时报告两套结果：固定 0.5 用于模型能力审计；校准阈值用于 Macro-F1 导向的部署。若业务目标是整组复合干扰完全识别，应直接在 calibration 上优化组合级或 subset 目标，而不是逐类别最大化 F1。

## 最终决策

1. 分类主干优先保留 `legacy_mlcnn`；`legacy_resnet34` 作为结构差异较大的次选对照。
2. 参数提取暂不因 Legacy 接口打通而接回正式 C2：当前无模型达到相对常数基线 NMAE 降低 10% 的门槛。
3. 参数方向下一步应固定 MLCNN 分类骨干，比较参数头输入：共享全局特征、浅层多尺度特征、类别条件特征；同时保留 IQ-Grouped 作为参数 NMAE 上界对照。
4. EfficientNet 暂停继续做纯学习率/weight decay 网格；若重启，应先验证小 batch 下 BatchNorm 统计、GroupNorm 替换或 BN 冻结，而不是扩大同类超参数搜索。
5. ConvNeXt 可保留 `warmup + Cosine, lr=3e-4, wd=1e-4` 作为现代骨干对照，但当前不应替代 MLCNN。

## 产物

- 现代骨干小搜索：`outputs/modern_tuning_screen/tuned_multitask_20260725_084329`
- EfficientNet 3-seed：`outputs/modern_tuning_confirm_effnet/tuned_multitask_20260725_084904`
- ConvNeXt 3-seed：`outputs/modern_tuning_confirm_convnext/tuned_multitask_20260725_085101`
- Legacy 全骨干单种子：`outputs/legacy_multitask_screen/tuned_multitask_20260725_085250`
- Legacy 前两名 3-seed：`outputs/legacy_multitask_confirm/tuned_multitask_20260725_085813`
- 标准骨干 3-seed：`outputs/standard_multitask_confirm/tuned_multitask_20260725_085953`
- 全模型 GPU smoke：`outputs/tuned_multitask_smoke/tuned_multitask_20260725_084246`

