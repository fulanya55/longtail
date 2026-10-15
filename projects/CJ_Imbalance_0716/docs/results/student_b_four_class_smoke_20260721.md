# 师弟 B：四类低标注 DAPS-CJNet-v2 首轮 GPU 冒烟记录

> 本记录只验证 20% labeled 条件下监督、半监督和参数感知路径是否真实运行。每组仅训练 2 epochs，且为了观察伪标签回传，B5/B6 的首轮阈值临时设为 0.5/0.5；这些数值不得用于论文结论。

## 已闭环的功能

- 5%、10%、20%、50%、100% 嵌套标注子集；四类 20% 对应 840 labeled 与 3360 hidden-label 样本。
- `HiddenLabelDataset` 不向 trainer 返回真实标签或真实参数。
- weak/strong 复数域增强均返回参数安全 mask；含附加噪声时 JNR 自动排除出参数一致性。
- EMA Teacher、逐标签正/负/不确定伪标签、稳定性门、prototype reliability、dynamic threshold。
- label-gated parameter heads；仅在 `label=1` 且 `parameter_mask=1` 的位置计算 Smooth L1。
- 无效参数位 NaN 的回归缺陷已通过“先索引有效位置再计算损失”修复并加入回归测试。

## 冒烟矩阵

| 变体 | 说明 | 训练参数 | 部署参数 | 最大伪标签 coverage | 非零无监督分类损失 | 非零特征一致性 | 非零监督参数损失 | 非零无监督参数损失 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 20% supervised-only | 6308 | 5684 | 0 | 否 | 否 | 否 | 否 |
| B5 | DAPS，无参数分支 | 6308 | 5684 | 1.0 | 是 | 是 | 否 | 否 |
| B6 | DAPS + 参数分支 | 8748 | 8748 | 1.0 | 是 | 是 | 是 | 是 |

B5/B6 在第 1 epoch 的伪标签 coverage 为 1.0；动态阈值根据 calibration precision 将正阈值提高后，第 2 epoch coverage 降为 0。这说明门控和阈值更新均实际生效，也说明正式实验不能使用 0.5/0.5 激进起点，应恢复 Prompt 预注册的 0.95/0.05，并用更长 warm-up 逐步提高 coverage。

## 论文实验使用方式

正式四类 pilot 先固定 20% labeled，按 B0→B6 顺序运行；每个变体必须复用相同 `split_sha256` 与 `semi_split_sha256`。第一轮只跑 seed 2026，确认无 NaN、伪标签 precision/coverage 合理后，再扩展 3 seeds。六类正式实验再使用 5%、10%、20%、50% 与 5 seeds。

```powershell
conda run -n YDS37 python experiments/run_student_b.py `
  --config configs/student_b/daps_param.yaml `
  --variant B6 --labeled-ratio 0.20 `
  --epochs 100 --batch-size 64 --amp
```

正式配置必须保留 `tau_pos=0.95`、`tau_neg=0.05`、teacher warm-up 和独立 calibration 更新；locked test 仍禁止读取。
