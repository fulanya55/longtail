# 路线 A：SawFM WeightFusion 固定权重新种子确认方案（2026-07-23）

## 1. 目标

上一阶段在 seeds 2041～2043 上选择并验证了：

- WeightFusion：alpha=0.2；
- LogitFusion：alpha=0.6。

由于这三个种子参与过 alpha 选择，本轮新增 seeds 2039、2040，在不重新搜索 alpha 的条件下确认融合收益是否能复现。

## 2. 正式协议

| 项目 | 设置 |
|---|---|
| 数据 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 配置 | `configs/student_a/six_class.yaml` |
| 新种子 | 2039、2040 |
| Label-LT | N=2400，SawFM=400，IF=4 |
| 训练 | 20 epochs，batch size 64，lr=3e-4 |
| 损失 | 完整 DB |
| 后期端点 | macro-AUPR 选点 |
| 早期端点 | SawFM-AUPR 选点 |
| WeightFusion | 固定 alpha=0.2 |
| LogitFusion | 固定 alpha=0.6 |
| locked test | 禁止使用 |

两个端点采用相同随机初始化和确定性训练轨迹，只允许检查点选择不同。逐 epoch `train_loss` 与 `macro_f1` 必须完全一致。

## 3. 数据角色

- early-stop：保存 DB-Macro 与 DB-Tail 端点；
- calibration：只做温度校准，不重新选择 alpha；
- threshold-selection：改作独立 fusion validation，不选择阈值；
- locked test：不读取。

## 4. 主门槛

主候选为 WeightFusion-0.2。新种子确认结果相对 DB-Macro 必须同时满足：

1. 平均 macro-AUPR 增量不低于 -0.005；
2. 平均 SawFM AUPR 增量不低于 +0.005；
3. 2/2 seeds 的 macro-AUPR 不低于 -0.005；
4. 2/2 seeds 的 SawFM AUPR 为正；
5. locked test 未使用。

LogitFusion-0.6 作为性能上界报告，不决定单模型主方法是否晋级。

## 5. 汇总边界

- seeds 2039、2040：纯确认集，是本轮主要结论；
- seeds 2041～2043：发现/内部验证集；
- 五种子合并结果：用于稳定性描述，但不能伪装成五个全部未参与调参的确认种子。

若 WeightFusion 通过，下一阶段进入 VFT、ISRJ 的固定 alpha 跨尾类筛选；若失败，则停止静态参数融合，转向训练期尾类保持约束。

## 6. 执行状态

已于 2026-07-23 按上述冻结协议完成。WeightFusion-0.2 的独立确认门槛未通过，正式决策为停止静态参数融合；完整数值、失败归因与后续条件残差蒸馏方向见 `docs/results/route_a_sawfm_weight_fusion_confirmatory_results_20260723.md`。
