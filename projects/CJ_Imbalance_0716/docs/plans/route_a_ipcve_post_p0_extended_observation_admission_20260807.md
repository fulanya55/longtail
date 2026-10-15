# Route A IP-CVE：P0 后扩展观测协议准入审计（预注册）

日期：2026-08-07  
证据角色：只读历史证据重叠审计；不生成数据、不训练模型、不校准、不选阈值、不访问任何测试样本。

## 1. 唯一问题

P0 已经否定当前单视图 B0 表征中的统一三目标 IP-CVE。下一步只允许改变观测协议或物理观测量。本审计只回答：

> 项目现有代码、数据生成能力和未消费身份条件下，是否存在一个不与既有失败线实质重叠、可直接进入新 H0 的本地扩展观测协议？

## 2. 固定候选族

审计只比较以下三个候选族，不在结果后增加第四个候选：

1. MFT 更长延迟孔径、LFM 匹配滤波或其低维残差；
2. VFT 多脉冲/交错 PRI、真实时间戳相干统计及其学习接回；
3. ISRJ 单通道包络自相关、周期一致性、块置乱控制或其学习接回。

## 3. 历史重叠判据

候选若满足任一条件即判为重叠并关闭：

- 观测量、输入和因果对照与已关闭实验相同，仅改变网络宽度、损失、阈值、统计区间或名称；
- 固定物理 H0 已通过，但同机制的容量匹配学习接回已失败；
- 弱目标相对 matched absent 的标签归因已经在两种独立协议中失败；
- 后续所需观测在当前 generator/inference contract 中不存在，必须新增传感器通道、受控接收机设置或真实/HIL 外部状态。

## 4. 新 H0 准入硬门

只有同时满足以下条件才允许生成新数据：

1. 推理期观测量与当前单路复 I/Q 实质不同；
2. 不读取 exact absent、pair id、功率、真参数或第二个反事实视图作为推理输入；
3. 对 MFT、VFT、ISRJ 三者都定义 label-attributable 的弱正/缺失零拟合对照；
4. 相对信息破坏控制具有预注册的效应量、置信区间和逐身份方向门；
5. 代码和数据协议能够真实产生该观测，而不是把训练标签或仿真 oracle 包装成传感器输入；
6. 新身份与历史 development、locked、test-v2、test-v3 全部隔离；
7. test-v3 保持 `access_authorized=false`、`analysis_allowed=false`、`used=false`。

任一硬门失败时，决策固定为 `no_run`，不得通过新增 adapter、attention、MLP、损失搜索、阈值放宽或重生成旧协议来挽救。

## 5. 固定输入证据

- IP-CVE P0 frozen observability summary；
- H0u/H1u/H2v MFT 延迟孔径链；
- H0q/H1a/H1b/H2 VFT 交错 PRI 链；
- P1/P2 全目标 H0 与 ISRJ 失败根因诊断；
- 上述协议的 active generator 代码，仅审计是否存在第二校准通道/第二 receiver/受控接收设置。

## 6. 固定产物

- `outputs/route_a_ipcve/post_p0_observation_admission/admission_audit.json`
- `outputs/route_a_ipcve/post_p0_observation_admission/run_receipt.json`
- `docs/reports/route_a_ipcve_post_p0_extended_observation_admission_20260807.md`
- `docs/results/route_a_ipcve_goal_negative_closure_20260807.md`

