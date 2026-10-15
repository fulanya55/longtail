# Route-A P2c：可见性辅助任务（2026-08-27）

## 目的

在 C1 纯六类和 F2 主干固定的前提下，使用数据中每个样本的 `weak_component_mask` 构造活动分量可见性目标。辅助目标只在活动标签上计算：强分量目标为 1，弱分量目标为 0；不存在的标签不参与辅助损失。比较 F2+BCE 与 F2+BCE+可见性辅助头。

## 协议

- 30 dB train-only 纯六类：4,096 拟合、2,048 验证；
- 各功率 train-only 未见身份开发：512 样本；
- 3 seeds：2039/2040/2041；2 epochs；batch=16；
- BCEWithLogitsLoss + AdamW（1e-4/1e-4）；
- 辅助权重 `lambda=0.1`；
- 未读取普通 test 或 locked test。

## 结果

| 模型 | 参数量 | 验证 Macro-AUPR | 验证组合准确率 | 10 dB AUPR | 20 dB AUPR | 40 dB AUPR | 45 dB AUPR | 50 dB AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F2+BCE | 258,541 | 0.998817 | 0.926595 | 0.857275 | 0.892107 | 0.927326 | 0.870694 | **0.823043** |
| F2+BCE+visibility | 258,541 | **0.998950** | **0.929199** | **0.858167** | **0.893160** | **0.929026** | **0.872408** | 0.822426 |

逐标签观察显示，辅助头主要改善低功率 MFT/ISRJ，以及高功率 SinFM；但高功率 AMA/FMB 和 SawFM 存在轻微波动，说明可见性信号并非对所有标签等价有效。

## 结论与门控

1. 可见性辅助任务对主分类有小幅正向证据，验证组合准确率提高约 0.26 个百分点。
2. 10–20 dB 和 40–45 dB 有轻微改善，但 50 dB 没有改善，尚不能称为完整功率鲁棒性解决方案。
3. 该结果为 3-seed、2-epoch、train-only 开发证据，判定为 **partial support**，不直接冻结。
4. 下一步应在保持辅助头的前提下，引入严格有界的困难负样本损失，并检查是否能减少 MFT/ISRJ 误报而不牺牲 AMA/SawFM。

## 工件

- [P2c 汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2c_visibility_aux_20260827\p2c_visibility_aux_summary.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2c_visibility_aux_20260827\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2c_visibility_aux_20260827\terminal_receipt.json)
- [执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p2c_visibility_aux.py)
