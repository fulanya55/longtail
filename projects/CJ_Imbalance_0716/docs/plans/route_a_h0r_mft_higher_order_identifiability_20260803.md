# Route A H0r：MFT 三阶相位耦合零拟合可辨识性预注册

状态：预注册；正式 source 尚未生成。

## 单一假设

在新的单标签 MFT 3-copy versus 8-copy 配对源中，固定 segmented bicoherence
（8 个等长段、DFT bins `(7,11)`）相对保持完整 DFT 幅度谱和圆周二阶自相关的
随机相位 surrogate，具有 MFT 副本数量的额外可辨识性。

## 数据与控制

- 新 identity pair seed：`3_800_000_000..3_800_000_639`；每对共享 LFM、delay
  候选集合、幅度 RNG 与 receiver-noise seed，只改变副本数 3/8。
- 每个 arm 以总 component JNR 20 dB 加入 proper-complex unit-variance receiver
  noise；不得使用复合背景、训练、校准、阈值或任何 test role。
- 对每条记录计算 full-record DFT phase surrogate；审计每条的 DFT magnitude 与
  circular-autocorrelation 最大相对误差。

## 固定统计、门槛与停止规则

统计、segment 数、bins、source size、JNR、seed、bootstrap draws `5000` 均固定。
主门为 candidate 3-vs-8 AUROC `>=0.65`、candidate-control AUROC `>=+0.10`、
paired bootstrap 95% CI 下界 `>0`、以及 candidate 8>3 paired fraction `>=0.65`。
控制误差门：全 source DFT magnitude relative error `<=1e-8`、circular autocorrelation
relative error `<=1e-8`。全部为零训练参数。

任一门失败即停止 H0r；不改 bins、segment、copy count、JNR、source size、surrogate、
统计、损失或网络。通过也只允许进入新的复合 H1，不能进入统一模型或 test-v3。
