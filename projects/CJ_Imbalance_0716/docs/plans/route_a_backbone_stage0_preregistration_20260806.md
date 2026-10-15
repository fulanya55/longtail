# Route A Backbone 强化 Stage 0 预注册

日期：2026-08-06  
状态：在任何正式 backbone 训练前冻结  
证据等级：数据准备与工程审计，不构成模型性能证据

## 1. 目标与边界

本阶段只建立全新的 `backbone-development-v1` train-only 数据源、五组
receiver-identity 隔离角色和 B0 工程合同。它不训练、选择或比较任何新
backbone，也不读取任何 test-v3 I/Q、标签、预测或指标。

- 项目：`E:\LJQ_code\CJ_Imbalance_0716`
- 生成协议：0716 六类闭集生成器，六类、63 个非空组合、每组合 400 条
- 输出：`E:\LJQ_code\干扰数据\route_a_backbone-development-v1_20260806\closedset_6class_train.mat`
- 只生成 train；不得构造 test manifest 或 test MAT
- Python：`D:\anaconda3\envs\PY10\python.exe`
- MATLAB：`D:\Matlab\bin\matlab.exe`
- test-v3：`access_authorized=false`、`analysis_allowed=false`、`used=false`

## 2. 身份预注册

- `cfg.seed = 2550000000`
- `sample_seed = [2550000001, 2550025200]`
- receiver 派生公式：
  `(sample_seed + 2*2246822519) mod 2^32`
- `receiver_noise_seed = [2748677743, 2748702942]`
- 样本数和 receiver identity 数均固定为 `25200`
- 当前安全合同扫描覆盖 `experiments/`、`docs/`、`outputs/tapc/`、
  `configs/` 中的 receiver range 记录，并排除任何 test-v3 命名路径；
  候选区间必须对全部登记区间零重合。

任何实际 MAT 的样本数、identity 唯一数、首尾区间或文件名合同不一致，
均立即停止，不能通过修改区间、删除旧登记或忽略冲突继续。

## 3. 五 seed 角色协议

固定训练 seed：`2039, 2040, 2041, 2042, 2043`。

每个 seed 先按六标签组合分层形成互斥的 optimization、early、calibration、
threshold 角色；再只在 optimization 中构造固定标签长尾：

- 目标标签计数：`[1600, 400, 1213, 919, 696, 528]`
- 目标样本数：`2400`
- fit/validation：对长尾样本固定随机顺序后按 `80%/20%` 切分
- early、calibration、threshold：各组合各 40 条，由分层角色直接提供
- 同一 receiver identity 不得跨任意两个角色
- 本数据源每个 receiver 只有一个视图；若后续增加多功率视图，必须按
  receiver identity 整组继承本次角色，禁止重新拆分视图

每个 seed 保存 `role_indices.npz`、角色样本/receiver hash、标签正例计数、
弱正计数、split hash 和长尾 manifest hash。

## 4. B0 固定工程合同

- 模型：`F2_DFMS_Full / dfms_iq / width=16`
- 输入：`[B,2,3000]`，`complex_rms`
- 输出：logits `[B,6]`，global feature `[B,128]`
- 参数量：`257511`
- stage shape：
  - stage1 `[B,16,3000]`
  - stage2 `[B,32,1500]`
  - stage3 `[B,48,750]`
  - stage4 late-fusion temporal `[B,128,375]`
- 本阶段不修改 F2 源码，不复用旧 F2 数值作为新来源 B0 结果

## 5. 后续训练冻结项

Stage 1 及 Stage 2 的 B0/B1-B4 共同采用：Focal
`gamma_pos=0, gamma_neg=3`，五个非 SawFM 标签均值加 `0.2*SawFM`，
AdamW，36 epochs，patience 8，batch 64，classifier LR `5e-4`，
backbone LR `5e-5`，weight decay `1e-4`，gradient clip `5.0`。
校准只用 calibration 的 per-label Platt；阈值只用 threshold，在
negative FPR 不超过 0.20 时最大化弱召回。

## 6. Stage 0 通过条件

1. MATLAB dry-run manifest 合同通过；
2. train-only MAT 原子生成并通过增强回载；
3. MAT SHA-256、生成器/审计脚本 SHA-256、identity registry hash 完整；
4. 六标签均存在弱正样本；
5. 五个 seed 的五角色样本与 receiver identity 均零重合；
6. 真实 MAT read、B0 forward/backward、checkpoint reload 和 CUDA smoke 通过；
7. test-v3 三个封存标志仍为 false；
8. 生成与审计均有明确 completed 或 aborted 终端收据。

Stage 0 未通过前不得启动 B0 正式五 seed 训练。
