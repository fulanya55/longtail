# Route A v15: PAM-IQ 位置感知标签查询（预注册）

## 1. 上一轮证据与单一可证伪假设

冻结 F2 的 v3 标签查询源诊断未通过：相对于全局 128 维特征，内容查询的五种子平均弱证据 AUROC 变化为负；因此不得把“增加标签查询”本身视为有效方案。与此同时，12 段 token 相对逐记录打乱仍有稳定信息，说明时序顺序存在但冻结 F2 没有把它转化为跨种子稳定的标签判别。

本轮唯一假设为：**若时序位置与局部证据必须在主干学习阶段共同适配，则在完全相同 PAM-IQ 主干和分类器中，把确定性绝对时间位置加入标签查询 key，可提升 VFT/ISRJ 弱正相对于负样本的排序，而不以参数增加或阈值放宽换取收益。**

这是端到端位置感知机制筛查，不是对冻结 F2 读出的重跑，也不是对已失败的 query 温度、宽度或学习率搜索。

## 2. 数据与不可访问边界

- 环境：`D:\anaconda3\envs\PY10\python.exe`。
- 训练数据：新生成且仅训练角色的 `复合干扰数据0802_v15_pamiq_positional_query`；基础种子 `450000000`，receiver-noise identity 范围 `[648677743, 648702942]`。
- 生成器仅请求 `train` split。不会生成、打开或登记 v15 test。
- 各模型的训练、early-stop、calibration、threshold 角色均按 `receiver_noise_seed` 身份隔离；九功率视图不作为独立跨角色样本。
- 历史 locked test、test-v2、v15 test 与封存 test-v3 一律禁止访问；所有 receipt 必须显式记录 `false`。

## 3. 完全匹配的网络对照

两组均采用端到端 PAM-IQ：真复数 I/Q stem、相位差增量、三阶段多尺度残差特征编码器、稀疏注意力、全局 mean/std 安全旁路、六标签 query 特征和 `GlobalResidualLabelQueryClassifier`。

| 组别 | 仅有差异 | 可训练参数 |
|---|---|---|
| Q0 ContentQuery | `LabelQueryPooling1d`，query 仅对内容 key 注意力 | 基准 |
| Q1 PositionalQuery | 在相同内容 key 上叠加固定正弦绝对位置编码 | 与 Q0 严格相同 |

固定参数：`width=16`、`query_dim=32`、`attention_floor=0.20`、`dropout=0.20`、类别数 6。位置编码没有可训练参数。单元测试将验证前向、反传、特征维度与参数数严格相同。

## 4. 训练、校准和阈值

- 五个训练种子：`2039..2043`；先单种子两 epoch smoke，再正式 20 epoch。
- `label_lt`：2400 样本、SawFM 锚定正样本 400、imbalance factor 4；同一 seed 下 Q0/Q1 使用同一 manifest 与角色 split。
- 损失：Distribution-Balanced loss；AdamW，学习率 `3e-4`、weight decay `1e-4`、batch 64、AMP。
- early stop 的选择量：macro-AUPR；正式运行耐心值等于训练轮数，避免以 early 结果调轮次。
- 温度校准只拟合 calibration role；每标签 shrinkage-F1 阈值只在 threshold role 拟合。主排序指标不依赖阈值；阈值指标只用于安全性审计。

## 5. 指标和晋级门槛

主指标为 Q1-Q0 的 VFT/ISRJ 平均“弱正 vs 标签负样本”AUROC（弱正定义：相对功率不高于 10 dB）。同时报告每类 AUPR、弱召回、负样本 FPR、macro-AUPR 和 macro-F1。

Q1 仅在以下条件**全部**通过时才进入一份新的外部确认数据设计：

1. 五种子每对的部署参数严格相同；
2. 目标平均弱 AUROC 平均增益至少 0.015，配对 bootstrap 95% CI 下界大于 0，且至少 4/5 种子为正；
3. macro-AUPR 不低于 Q0 超过 0.005；VFT、ISRJ AUPR 各自不低于超过 0.010；
4. 目标平均 FPR 不比 Q0 增加超过 0.005，且 Q1 的绝对平均 FPR 不超过 0.20。

任何一条失败：停止该位置查询机制，不再针对位置编码频率、query 温度、宽度、学习率或 epoch 调参；记录失败原因并回到新的、独立的物理机制假设。通过也不允许访问 test-v3。
