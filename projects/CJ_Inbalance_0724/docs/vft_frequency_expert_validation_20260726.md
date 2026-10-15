# VFT 瞬时频率序列专家改进与验证报告

## 1. 目标和冻结边界

上一阶段已经冻结复数 STFT U-Net，并通过 ISRJ 周期专家将三类系统的 tune Macro Recall 从 0.6384 提高到 0.6986。剩余主要短板是 VFT：U-Net 原始 VFT 头三 seed 平均 Recall 为 0.6714，且 seed 2042、2043 分别只有 0.6266、0.6125。

本轮保持 U-Net、SawFM 输出和 ISRJ 周期头不变，只增加 VFT 瞬时频率序列专家。train 用于拟合频率分类器，early 用于拟合冻结分数堆叠器，calibration 在 FPR 0.10 上限下确定阈值，tune 用于候选选择。confirm 和 locked test 均未读取。

预注册晋级条件为：VFT 平均 Recall 不低于 0.70、平均 FPR 不高于 0.10、Recall 至少提升 0.05，并且 3/3 seeds 同方向。

## 2. 方法

### 2.1 输入和瞬时频率序列

输入仍为当前 10 dB 单视图 `[B,2,3000]` I/Q。首先通过冻结 U-Net 得到扫频分量估计，再计算相邻复采样的相位增量：

`f_t = angle(x[t+1] * conj(x[t])) / pi`。

相位增量用局部幅度可靠度加权，并计算 9 点、33 点平滑轨迹及一阶变化率。所有特征仅使用时域顺序和局部池化，不使用全局 FFT。

### 2.2 候选专家

- trajectory：原始、9 点平滑、33 点平滑瞬时频率的 60 段均值和标准差；
- dynamics：瞬时频率局部极值、变化率和幅度轨迹；
- combined：trajectory 与 dynamics 拼接；
- 分类器：HistGradientBoosting leaf-7/15 或标准化 LogisticRegression；
- 融合器：early 集拟合 LogisticRegression，将冻结 U-Net VFT logit 与频率专家分数进行堆叠。

按预注册的“满足 FPR/AUPR 约束后最大化平均 Recall”规则，最终选择 `stack_trajectory_leaf15`。

## 3. 三 seed 结果

| VFT 方法 | AUPR | Recall | FPR | Recall 增量 | 同向 seeds |
|---|---:|---:|---:|---:|---:|
| 冻结 U-Net 原分类头 | 0.9053 | 0.6714 | 0.0952 | - | - |
| trajectory leaf-7 堆叠 | 0.9685 | 0.9047 | 0.0968 | +0.2333 | 3/3 |
| trajectory leaf-15 堆叠 | **0.9734** | **0.9245** | **0.1000** | **+0.2531** | **3/3** |
| 全特征堆叠 | 0.9777 | 0.9396 | 0.1027 | +0.2682 | 3/3 |

全特征堆叠虽然 Recall 更高，但平均 FPR 超过 0.10，因此未被选择。最终 leaf-15 轨迹专家满足全部晋级条件。其单 seed FPR 最大为 0.1081，说明 calibration 阈值迁移到 tune 后仍有小幅波动；平均 FPR 恰为 0.1000，后续不能再放宽阈值。

## 4. 接回三类系统

系统保持 SawFM 为 U-Net 输出、ISRJ 为上一阶段 `stack_all_periodicity`，只替换 VFT 输出。

| 系统 | Macro AUPR | Macro Recall | Macro FPR |
|---|---:|---:|---:|
| U-Net + ISRJ 周期头 | 0.9182 | 0.6986 | 0.0937 |
| U-Net + ISRJ 周期头 + VFT 频率头 | **0.9409** | **0.7830** | **0.0953** |

最终三类平均 Recall：SawFM 0.7031、VFT 0.9245、ISRJ 0.7214。三个 seed 的系统 Macro Recall 分别为 0.7885、0.7646、0.7958，均超过 0.70；系统 Macro FPR 分别为 0.0989、0.0941、0.0930。

因此，本轮在开发角色上通过整体 promotion gate。

## 5. 机制与泄漏对照

在 seed 2041 上使用同一固定 leaf-15 分类器完成以下对照：

| 输入/对照 | Tune AUPR | Recall | FPR |
|---|---:|---:|---:|
| 原始波形瞬时频率轨迹 | 0.8374 | 0.4547 | 0.0710 |
| U-Net 分离波形瞬时频率轨迹 | **0.9828** | **0.9625** | 0.1065 |
| 分离波形全局统计 | 0.9362 | 0.7859 | 0.0919 |
| 分离波形、训练标签打乱 | 0.5561 | 0.1859 | 0.1194 |

结论：

- 标签打乱后性能接近类别先验，未发现标签泄漏；
- 分离波形相对原始波形 AUPR 提高约 0.145，证明 U-Net 分离是必要环节；
- 时间分辨轨迹相对全局统计 AUPR 提高约 0.047、Recall 提高约 0.177，验证了瞬时频率序列归纳偏置；
- VFT 提升不是通过全局 FFT 或简单放宽阈值取得。

## 6. 当前证据边界与下一步

方法在 tune 开发角色上已经达到预定门槛，但这仍是查看过历史 confirm 后开展的自适应开发结果，不能把旧 confirm 再作为无偏确认集。本轮未使用 locked test。

建议现在停止架构和阈值调优，冻结：

1. 复数 STFT U-Net checkpoint 规则；
2. SawFM 原 U-Net 输出；
3. VFT `stack_trajectory_leaf15`；
4. ISRJ `stack_all_periodicity`；
5. calibration FPR=0.10 阈值规则及全部随机种子。

冻结并生成最终模型/阈值/代码哈希后，可按既定治理规则申请一次性 locked test。未经明确授权，不应自动读取 locked test。

## 7. 审计产物

- VFT 专家实现：`cj0724/vft_separator_frequency_head.py`
- 机制对照：`cj0724/vft_frequency_mechanism_controls.py`
- 三 seed 聚合：`outputs/vft_separator_frequency_head_20260726/summary.json`
- 机制对照结果：`outputs/vft_frequency_mechanism_controls_20260726/result.json`
- 协议记录：`outputs/vft_separator_frequency_head_20260726/protocol.json`
- 自动化测试：`70 passed`

所有结果均记录 `confirm_used=false`、`locked_test_used=false`。
