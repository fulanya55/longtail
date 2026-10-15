# 师弟 A：四类长尾与弱成分路线首轮 GPU 冒烟记录

> 本记录只验证训练链路、消融开关、身份配对和损失回传是否闭环。每组仅训练 2 epochs、256 个优化角色样本，不能作为论文性能结论。

## 固定条件

- 类别：FMB、SinFM、MFT、ISRJ；严格排除含 SawFM/VFT 的样本。
- 输入：长度 3000 的一维复数 I/Q，按复数 RMS 归一化。
- 骨干：`complex_1d_true`，width=8；分类损失为 CB-ASL。
- 随机种子：2026；early-stop 角色最多取 128 个样本。
- A3/A4 功率视图：10/30/50 dB；参考等级 30 dB。
- 功率视图通过 `receiver_noise_seed` 与优化角色逐身份绑定，未重新划分。

## 最小消融矩阵

| 变体 | 标签原型 | 功率一致性 | 配对记录 | 最后一轮原型损失 | 最后一轮功率损失 | Best subset accuracy | Best Hamming loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 否 | 否 | 0 | 0 | 0 | 0.0781 | 0.4727 |
| A2 | 是 | 否 | 0 | 0.6230 | 0 | 0.0703 | 0.4707 |
| A3 | 否 | 是 | 512 | 0 | 0.0717 | 0.1094 | 0.4707 |
| A4 | 是 | 是 | 512 | 0.5970 | 0.0921 | 0.0625 | 0.4336 |

上述数值只说明四条路径均真实参与计算；尤其 A2/A4 的 prototype loss 与 A3/A4 的 power loss 均为非零。正式比较必须按 Prompt 使用完整优化集、5 个随机种子、校准角色和阈值选择角色，并报告均值与标准差。

## 可复现命令模板

```powershell
conda run -n YDS37 python experiments/run_student_a.py `
  --config configs/student_a/four_class.yaml `
  --variant A4 --loss cb_asl --epochs 60 --batch-size 64 --amp
```

依次将 `--variant` 改为 `A0`、`A2`、`A3`、`A4` 即可形成最小消融闭环。训练时保存完整模型与原型记忆；部署参数统计只计算原始 backbone/logits 路径，标签关系分支可在导出时移除。
