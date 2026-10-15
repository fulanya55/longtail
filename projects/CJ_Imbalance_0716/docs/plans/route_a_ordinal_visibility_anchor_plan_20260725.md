# 路线 A：同身份有序可见度坐标与三锚点校准方案（2026-07-25）

## 1. 已验证问题

上一轮22参数可见度条件仿射校准器将40/45 dB召回提高到0.710/0.540，但50 dB只有0.167。50 dB oracle recall仍为0.701，说明稳定S2保留了排序信息，失败来自校准器不能从单视图BCE中可靠推断极弱可见度等级。threshold role和early-development均出现相同失败，排除简单开发过拟合。

本轮显式利用同一 `receiver_noise_seed` 的 root/40/45/50 dB 视图，学习一个不读取真实功率数值的可见度坐标，并通过三个共享锚点产生校准偏置。F2、H2、稳定S2和C2 alpha=.80继续冻结。

## 2. 数据角色

沿用上一轮严格协议：

- 种子2041/2042使用已冻结S2；2043使用validation-only选定的初始化274302。
- 每种子从train role排除专家训练与普通路由身份，再按28组合×30对拆成expert-scale、visibility-fit、power-threshold三个互斥身份角色。
- visibility-fit中的SawFM正身份使用root/40/45/50 dB四视图构成有序组；hard negatives参与分类校准但不赋予SawFM可见度顺序。
- power-threshold用于checkpoint和统一阈值选择。
- early-development普通域与40/45/50 dB仅作最终评价。
- 多功率视图按identity分组，locked test冻结。

## 3. 模型结构

输入仍为上一轮10维内部统计，不输入JNR、功率档、raw RMS：C2/H2/S2 logits、分数差、其他标签logits、evidence top-k/max、stage2幅度均值与时间标准差。

### 3.1 可见度坐标

`v = Linear(10,1)`，共11参数。

训练阶段只使用顺序关系：

`v(root) > v(40) > v(45) > v(50)`

对相邻与跨级视图使用softplus pairwise ranking loss，margin=0.30。真实功率只用于构造顺序标签，不作为输入。

### 3.2 三锚点偏置

固定锚点中心为高/中/低可见度 `[1,0,-1]`。根据v到三个中心的距离进行soft assignment。三个锚点偏置被约束为：高可见度0，中可见度为正增量，低可见度为更大正增量。

最终logit：

`z' = positive_scale * z_C2 + global_bias + anchor_bias(v)`

总参数16：visibility linear 11、两个有序锚点增量、全局scale/bias、anchor temperature。参数量小于上一轮V2的22。

## 4. 对照

- `A0_PreviousV2`：上一轮22参数条件仿射结果，冻结引用。
- `A1_AnchorNoOrdinal`：相同16参数三锚点结构，只用分类BCE，不加有序损失。
- `A2_OrdinalAnchor`：BCE + 同身份有序可见度损失，唯一晋级候选。
- `A3_OraclePowerAnchor`：训练和推理时用真实root/40/45/50等级直接指定锚点，仅作为结构上限，不能晋级。

## 5. 训练与选择

- AdamW，学习率1e-2，weight decay1e-3。
- 最多300 epoch，patience50，梯度裁剪5。
- A2 ordinal weight=1.0，margin=0.30；A1 ordinal weight=0。
- 分类BCE按普通/root/40/45/50域等权。
- checkpoint选择在power-threshold及普通threshold role上，在所有FPR≤0.20条件下优先最大化三档平均recall、最差档recall、普通F1、再最小参数范数。
- 所有方法最终只能使用一个统一阈值。

## 6. 双门槛

### 可见度机制门槛

- A2在power-threshold同身份四视图的六对pairwise排序准确率≥0.75；
- early-development排序准确率≥0.70；
- 对应Kendall tau≥0.40；
- 至少2/3种子通过；
- A2排序准确率比A1提高≥0.10。

### 决策门槛

- 40/45/50 dB平均recall≥0.50，每档≥0.35；
- 每档hard-negative FPR≤0.20；
- 普通FPR相对A0增量≤0.02；
- 普通recall相对A0下降≤0.05；
- 至少2/3种子通过；
- 平均oracle gap≤0.15；
- A2三档平均recall不低于A0和A1；
- A3若仍无法达到平均recall0.60，则判定三锚点结构上限不足；
- 身份和checkpoint选择审计通过，locked test未使用。

机制与决策门槛必须同时通过，才允许解冻最后stage。

## 7. 失败后的决策

- A2机制通过、决策失败且A3通过：可见度坐标有效，但锚点到概率的映射不足，下一步只改映射。
- A2机制失败、A3通过：内部统计无法稳定预测可见度，转向stage2时序图上的专用visibility head。
- A3失败：三锚点统一阈值结构不够，考虑显式unknown/reject而非继续校准。
- 全部通过：执行最后stage小学习率联合微调，并保留冻结A2对照。

## 8. 产物

保存每种子三模型checkpoint/history、visibility scores、同身份pairwise/Kendall指标、阈值搜索、普通/功率预测、oracle gap、源checkpoint与identity hash、receipts和总决策。
