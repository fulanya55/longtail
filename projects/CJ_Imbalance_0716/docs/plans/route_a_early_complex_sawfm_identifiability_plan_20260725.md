# 路线 A：早期复数 SawFM 证据源可辨识性筛选（2026-07-25）

## 目标

上一阶段表明，冻结 Stage2 上的可靠性读出没有超过已有 visibility 统计，因此本轮不再预测“A0 是否出错”，而是直接检验更早证据源能否区分：

> SawFM 存在但在 40/45/50 dB 下变弱的样本，与其他标签组合相同但 SawFM 缺失的 hard negative。

本轮是 oracle 可辨识性筛选，不是生产模型训练。任何候选在独立 development 达到预注册门槛之前，不接回六类分类器、不解冻 F2、不使用 locked test。

## 冻结来源与角色

- 环境：`D:\anaconda3\envs\PY10\python.exe`
- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`
- 冻结模型：F2、H2、S2、C2/A0 全部只读
- 种子：2041、2042、2043
- 功率：40、45、50 dB
- fit：既有 `fit` 组合匹配正负身份
- validation：既有 `threshold` 组合匹配正负身份
- independent development：既有 development 身份；普通域仅作诊断，功率域为主评价
- 身份键：`receiver_noise_seed`
- locked test：不使用

## 候选证据源

所有输入均来自同一个冻结 F2 checkpoint，且数据仍使用项目的 RMS 归一化，避免总功率捷径。

| 代号 | 输入 | 作用 |
|---|---|---|
| P0 | H2 SawFM logit | 当前二元基线 |
| P1 | 10维 visibility 统计 | 上一阶段最强统计基线 |
| P2 | 归一化原始 I/Q 物理时序 | 检验网络前是否已有可分证据 |
| P3 | 第一复数残差层输出 | 主候选：检验早期复数特征是否优于 Stage2 |
| P4 | Stage2 复数输出 | 与既有 S2 结果对照 |
| P5 | Stage3 复数输出 | 检验更深压缩是否继续损失弱证据 |
| P6 | 原始 I/Q + 第一复数层 | 预注册的完整候选 |

固定物理描述包含：幅度、幅度差分、归一化相位增量、25段局部 mean/max/std、全局 mean/std/max/top-k。P1--P6 均使用 L2 Logistic Regression，`C={0.01,0.1,1,10}` 仅由 validation 选择，fit 内采用域/类别平衡权重。

## 主指标与晋级门

主指标是 independent development 的 40/45/50 dB：

- 各功率 AUROC/AUPR；
- mean-power AUROC；
- worst-power AUROC；
- validation 到 development 的 mean-power AUROC 落差。

P6 只有在至少 2/3 个种子同时满足以下条件才晋级：

1. mean-power AUROC `>=0.65`；
2. worst-power AUROC `>=0.60`；
3. 相对 P1 mean-power AUROC 增量 `>=0.03`；
4. validation→development 落差 `<=0.15`；
5. 50 dB AUROC `>=0.60`；
6. 身份重叠为 0、locked test 未使用。

若 P3/P6 达到可辨识性门但 P6 未满足完整晋级门，只允许在新 development seeds 上做固定结构确认；若所有早期候选均不超过 P1，则停止从冻结 F2 中继续寻找 SawFM 分支，下一步必须回到数据监督或独立前端，而不是继续加读出头。

## 产物

- 程序：`experiments/run_route_a_early_complex_sawfm_identifiability.py`
- 测试：`tests/test_early_complex_sawfm_identifiability.py`
- 汇总：`outputs/tapc/route_a_early_complex_sawfm_identifiability/route_a_early_complex_sawfm_identifiability_summary.json`
- 结果：`docs/results/route_a_early_complex_sawfm_identifiability_results_20260725.md`
