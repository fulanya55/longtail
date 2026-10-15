# 直接新六类 IF3 数据集 development-v3：Stage 0 审计

## 结论

本轮按既有 Goal 计划尝试直接使用：

`E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA`

但 Stage 0 未通过“全新身份 development-v3”门槛，因此未启动模型训练。

## 已核对事实

- 类别：FMB、AMA、SawFM、SinFM、MFT、ISRJ；
- train/test：各 8000 个样本；
- IF=3.0、CF=7.8537、63 个组合完整；
- train SHA-256：`dd7ee3f823e88174e347dca6e57d2d49c48d9d30f8c86c7ea707eefbe3411da6`；
- test SHA-256：`360e5e682df279902df550336bc6f008c46b459e6c5efc51aaf225a8f1066e52`。

## 阻断原因

上一轮直接六类基线已经把 train 文件完整划分为：

- fit：6415；
- validation：1585；
- 合计：8000。

因此无法从同一个 train 文件中再构造既不复用旧身份、又保持 IF3/CF 和样本预算一致的新 development-v3。该数据集的 test 文件也已经被用于开发测试和校准诊断，不能重新用于阈值选择或模型调参。

如果强行重分 train 并运行，只能称为“非新身份的直接重复实验”，不能称为 development-v3 五种子确认。

## 当前状态

- Stage 0：`blocked_for_fresh_identity_development_v3`；
- Stage A 2×2：未执行；
- Stage B/C/D：未执行；
- locked test：未读取；
- E5：未修改。

## 可执行的下一步

1. 从原始九类池重新生成一个新的六类 IF3 数据集，使用新的 receiver/noise seed，保持 IF3、CF8 和 63 组合协议；
2. 或明确授权进行非新身份的直接复现实验，并在报告中标为 `re-evaluation / consumed-development-data`，不作为新确认结论。

本次阻断收据：

[stage0_audit.json](../../outputs/route_a_ama_direct_sixclass_development_v3_20260905/stage0_audit.json)
