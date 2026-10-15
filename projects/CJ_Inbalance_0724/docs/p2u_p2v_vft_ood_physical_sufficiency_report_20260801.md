# P2u/P2v VFT Doppler 参数 OOD 物理充分性报告

## 决策

P2u 和 P2v 均未通过预注册物理充分性门禁。依据“Oracle 充分性失败时不增加深度路由容量”的规则，当前停止 VFT 宽域网格与相位增量两条机制线，不进入新种子确认，也不挂接完整 12 参数系统。

全部试验仅使用新生成的 `train` 角色数据；`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## P2u：有符号宽网格

P2u 对比固定常数、冻结 `1--7 kHz` 正频网格、`-30--10 kHz` 宽有符号网格、错误 LFM 参考和无时延对齐。

宽网格在负 Doppler OOD 上证明了方向性物理价值：

- far-negative：MAE `8944.9 -> 3510.3 Hz`，Pearson `-0.030 -> 0.322`；
- far-negative-high：MAE `23599.1 -> 9391.4 Hz`。

但它严重破坏正频区域：

- ID-control：MAE `459.4 -> 4393.0 Hz`，Pearson `0.704 -> 0.234`；
- near-low：MAE `3980.8 -> 4277.2 Hz`；
- near-high：MAE `3481.5 -> 5169.3 Hz`。

根因是网格范围扩大后，单次全域 argmax 的极值偏差和伪峰机会显著增加。错误参考与无时延对齐更差，证明参考对齐本身有效，但不能消除宽域多重比较风险。

## P2v：固定 64 点相位增量

P2v 使用全新身份种子 `3600000000`，630 个组合均衡身份、63 个非空组合、25/35/50 dB，且与 P2i--P2u 代表身份文件重叠为 0。

合成纯 VFT 信号上，参考对齐后的 64 点相位增量可以恢复 `+4.5 kHz` 和 `-22 kHz`。但在真实复合干扰数据中失败：

- ID-control MAE `11582.4 Hz`，Pearson `0.101`；
- near-low MAE `11897.1 Hz`，Pearson `0.053`；
- near-high MAE `11459.7 Hz`，Pearson `0.015`；
- far-negative MAE `12199.9 Hz`，Pearson `0.107`；
- far-negative-high MAE `13556.6 Hz`，Pearson `0.159`。

五区间平均相干度都集中在约 `0.615--0.617`，没有形成区间或正确性分离。这表明复合干扰、LFM 时延--Doppler 耦合和成分混合共同破坏了直接相位斜率估计。

## Oracle 审计

诊断阶段逐样本使用真实误差在冻结正频网格、相位估计以及常数回退中选择最优者。该 Oracle 仅用于判断理论上限，不用于晋级。

- ID-control Oracle 可达到 Pearson `0.931`；
- far-negative 和 far-negative-high Oracle 分别达到 Pearson `0.379/0.333`；
- near-low 和 near-high 即使使用标签 Oracle，Pearson 仍为 `-0.220/-0.123`。

因此 `oracle_C1_C2_sufficient=false`、`oracle_C0_C1_C2_sufficient=false`。近正频边界缺少可路由的候选估计器，增加路由网络无法解决问题。

相干度对 `|error| >= 5 kHz` 的风险识别 AUPR 为 `0.720--0.782`，说明它可以作为未来“拒绝/回退”特征，但在当前候选专家集合 Oracle 不充分的情况下，不应构造深度路由器。

## 创新证据边界

当前可以成立的有限主张：

> 参考对齐的有符号搜索能显著改善负 Doppler OOD，并且错误参考、无时延对齐对照证明了 LFM 参考和时延对齐的必要性；但单次宽域搜索和直接相位增量均不能在正负全域保持统一充分性。

当前不能成立的主张：

> 已获得可部署、全范围泛化的 VFT Doppler 物理专家。

下一次恢复 VFT 路线必须先引入新的独立物理观测，例如双斜率/上下扫频参考、显式 VFT 成分分离或可观测的时延参数，而不是继续调整路由器阈值。

## 哈希

- P2u 预注册：`aa4e3b1d2ef84a0aaf78fc796d339991f6943f5ef3808f1dcd0f55ba211500f3`
- P2u 汇总：`ca7e8f9b809d18a0c1eafbfbcb736a503c90da8bbd43aaab1f8441fe18e9a4989`
- P2v 预注册：`2fcb9597e48c00e3912942157fc7c5aed965c0b6cb80d64b092a2a6ae2a4fe3a`
- P2v 实现：`e85d4e809eafbbd973ea1a69fe54a8341640f63a2361f996ad6c52c1c1d47e25`
- P2v 汇总：`da8ec017046b000140f39e282d50029a62911c68d152e8056fa456e1eda72c53`
- P2v Oracle：`c14279cd0b0846b99b41e894362ac10fe3c677533d1c1889b2c9cca9c06c3542`
