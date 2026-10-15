# 路线 A：可训练 Raw+Stage1 SawFM 小前端方案（2026-07-25）

## 1. 本轮问题

固定描述符已经证明 Raw I/Q + Stage1 对弱 SawFM 具有稳定可辨识性。本轮只回答：

> 一个小容量、可部署、端到端可训练的时序前端，能否复现固定 948 维 oracle probe 的主要收益？

本轮不接回六类分类器，不调整 H2 阈值，不解冻 F2，不使用 locked test。

## 2. 网络结构

输入仍为 RMS 归一化 `[B,2,3000]` I/Q。F2 第一复数残差层冻结并输出 `[B,16,3000]` 实部与虚部。

每个分支先构造六条无参数物理时序：

1. 通道平均幅度；
2. 通道最大幅度；
3. 通道幅度标准差；
4. 平均幅度差分；
5. 平均相位增量余弦；
6. 平均相位增量正弦。

可训练分支结构：

```text
[B,6,3000]
→ 1x1 Conv 6→16 + GroupNorm + SiLU
→ 三个 depthwise temporal branches, kernels 7/31/63
→ concat 48→16, 1x1 Conv + GroupNorm + SiLU
→ AvgPool stride 4
→ mean/std/max/top-k statistics: 64维
→ 与原始六时序安全统计 24维拼接
→ 每分支 88维
```

Raw+Stage1 将两个 88 维分支拼接为 176 维，再经：

```text
Linear 176→32 → GELU → Dropout 0.1 → Linear 32→1
```

对照：

- N0 `RawOnly`：只启用 Raw 分支；
- N1 `Stage1Only`：只启用 Stage1 分支；
- N2 `RawStage1`：两个分支融合。

## 3. 数据角色

完全复现已审计的身份角色：

- fit：训练网络；
- threshold：按 mean-power AUROC 选择 checkpoint；
- scale：未参与训练/选模的身份确认；
- independent development：第二个独立评价角色；
- 域：root、40、45、50 dB；
- 正负样本：其他五类组合匹配的 SawFM present / absent；
- `receiver_noise_seed` 不跨角色；
- locked test 不使用。

## 4. 损失与超参数

- 损失：四域等量样本上的 `BCEWithLogitsLoss`；
- 优化器：AdamW；
- 学习率：`3e-4`；
- weight decay：`1e-3`；
- batch size：64；
- 最大 epoch：40；
- patience：8；
- gradient clip：5；
- checkpoint：threshold 角色 40/45/50 mean AUROC 优先，mean AUPR 次优；
- 种子：2041、2042、2043；
- 参数规模目标：N2 小于 20k。

## 5. 预注册晋级规则

### 可部署前端门

N2 至少 2/3 种子满足：

1. development mean-power AUROC `>=0.90`；
2. development worst-power AUROC `>=0.85`；
3. development 50 dB AUROC `>=0.85`；
4. scale mean-power AUROC `>=0.90`；
5. threshold→development AUROC 落差 `<=0.15`；
6. 参数量 `<20k`；
7. locked test 未使用且身份审计通过。

### Stage1 互补门

只有 N2 相对 N0 的 development mean-power AUROC 在至少 2/3 种子提高 `>=0.01`，才保留 Stage1 分支；否则即使 N2 有效，也将 Raw-only 作为后续更低风险结构。

### 下一决策

- N2 通过且 Stage1 互补：固定 Raw+Stage1，在新模型 seeds 上确认；
- N0 通过但 Stage1 不互补：固定 Raw-only，在新模型 seeds 上确认；
- 均不通过：固定描述符的高分不能转化为小网络，停止该前端路线。

## 6. 产物

- 模型：`cjlt/models/evidence_readout.py`
- 程序：`experiments/run_route_a_trainable_raw_stage1_sawfm_frontend.py`
- 测试：`tests/test_trainable_raw_stage1_sawfm_frontend.py`
- 输出：`outputs/tapc/route_a_trainable_raw_stage1_sawfm_frontend/`
- 结果：`docs/results/route_a_trainable_raw_stage1_sawfm_frontend_results_20260725.md`
