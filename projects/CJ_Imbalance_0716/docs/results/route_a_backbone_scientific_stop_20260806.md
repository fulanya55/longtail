# Route A Backbone 强化 Goal：完整结果与科学停止结论

日期：2026-08-06  
数据：全新 receiver-identity 隔离 `backbone-development-v1`  
最终状态：`completed_scientific_stop`  
独立测试边界：test-v2、locked test、test-v3 均未使用；test-v3 继续密封。

## 1. Goal 是否完成

Goal 规定两种合法终点：冻结一个通过全部因果链的强化 backbone，或 M1–M4 均失败后形成完整科学停止。本轮完成了后者。

已完成：数据/身份审计、B0 五 seed 复现、M1–M4 独立预注册、实现、单元与真实 MAT/CUDA smoke、五 seed 正式训练、Platt、FPR 约束阈值、early 报告、配对 bootstrap、逐标签诊断、成本收据、失败关闭和总哈希清单。  
未执行且不应执行：Stage-3 同容量/信息破坏控制、失败模块组合、冻结 probe、全新身份 confirmation。原因是没有任何候选通过相对 B0 的 Stage-2 入场门槛。

## 2. B0 与四个候选

B0 development-only 基线：macro-AUPR `0.843746`，三目标弱正 AUROC `0.555864`，弱正 recall `0.237524`，强正 recall `0.771434`，FPR `0.195215`。

| 候选 | 唯一机制 | Δmacro-AUPR | Δ弱正 AUROC | Δ弱正 recall | Δ强正 recall | 绝对 FPR | 决策 |
|---|---|---:|---:|---:|---:|---:|---|
| M1 | 固定复数抗混叠下采样 | +0.004655 | -0.000354 | +0.002794 | -0.000119 | 0.194677 | fail/close |
| M2 | Cross-Stage Complex Feature Pyramid | -0.000002 | +0.000447 | +0.001075 | -0.000259 | 0.193871 | fail/close |
| M3 | 超长感受野 EPB | -0.000375 | +0.006936 | +0.008218 | +0.005884 | 0.191828 | fail/close |
| M4 | max + top-16 local statistics | -0.000404 | -0.000018 | -0.003301 | -0.004142 | 0.193871 | fail/close |

四个候选均远未达到弱正 AUROC `+0.020` 和弱 recall `+0.025` 的主要效应量门槛。M1 的总体 AUPR 小增益来自 FMB/SawFM 重分配，并稳定损害 SinFM/VFT；M2 的跨层残差被优化器压到约 ±0.004–0.007；M3 对 VFT/ISRJ 有标签特定弱证据收益，但显著损害 FMB；M4 在 4/5 seed 学成负 local residual，并使弱 recall 5/5 下降。

## 3. 训练充分性判断

未触发统一 72-epoch/cosine 审计。25 个正式 candidate/seed 运行中，多数最佳点位于 33–36，但最后点与最佳 validation common macro-AUPR 的差通常小于 0.005，末五轮斜率量级约 `1e-4–1e-3`，且 B0/M2 的最佳 epoch 模式几乎相同。没有“候选明显未收敛且延长训练可能改变机制结论”的充分证据。按预注册也禁止只延长失败候选或有利 seed。

## 4. 对当前网络缺陷的更新认识

本轮否定了四个简单解释：

1. 不是仅靠抗混叠就能恢复弱分量；频谱平滑会在调制类型之间重新分配收益。
2. 不是仅靠读取更早层就能恢复弱分量；跨层证据在当前目标下几乎没有被使用。
3. 更大局部感受野具有标签特定作用，但不是统一 backbone 改善，并会牺牲全局/强证据类型。
4. 直接加入局部极值统计不是答案；局部大响应多数仍由强分量或混杂结构主导。

因此，在当前合成观测协议、Focal 监督和共享全局读取方式下，继续增加 backbone 容量或局部统计并不是可可靠修复的主瓶颈。更可能的限制位于观测可辨识性与监督定义：弱标签的局部证据没有被明确地与同组合 hard negative、同 receiver 高低功率视图或真实测量域约束起来。

## 5. 科学停止后的允许方向

同一 development-v1 上禁止继续：扩宽 backbone、增加注意力、重调阈值、换 M1 核、提高 M2/M4 残差、修改 M3 dilation、组合失败模块或再次筛选候选。

后续研究只能另立新 Goal，并选择以下之一：

1. 新观测协议：成对采集同 receiver identity 的高/低功率视图，并提供成分级时间/频率可见性或分离监督；重点验证“可辨识性”而不是再堆主干。
2. 真实或半实物/HIL 数据：检验当前合成数据中的局部强响应与真实弱干扰证据是否同分布。
3. 论文范围收缩：将 M3 描述为 VFT/ISRJ 标签特定局部证据现象和负结果，不宣称统一强化 backbone；B0 继续作为六类主要基线。

若未来建立新观测协议，必须使用全新 identity 来源和新的预注册，不能回到本 development-v1 选择结构。当前结果不授权打开 test-v3。

## 6. 主要工件

- 数据与身份审计：`outputs/tapc/route_a_backbone_development_v1/identity_role_audit.json`
- B0：`outputs/tapc/route_a_backbone_b0_development_v1/b0_summary.json`
- M1–M4：各正式目录中的 summary、comparison、五 seed history/checkpoint/calibration/threshold/predictions 与 terminal receipt。
- M4 选型：`outputs/tapc/route_a_backbone_m4_selection/selection_receipt.json`
- 总清单：`outputs/tapc/route_a_backbone_scientific_stop_20260806/scientific_stop_manifest.json`

所有失败均作为正式负结果保留，没有用 test 或 early 结果返调结构。
