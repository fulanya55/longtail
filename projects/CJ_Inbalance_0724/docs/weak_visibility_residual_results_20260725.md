# 路线B弱分量可见性残差改进与实验验证（2026-07-25）

## 1. 结论

本轮验证了一个新的、可部署的弱分量分类机制：冻结现有 MLCNN 分类骨干和 IQ-Grouped 参数分支，仅增加轻量 IQ-Grouped 可见性残差头。残差头推理时只读取当前 `[B,2,3000]` I/Q 与当前视图的冻结分类 logit；30 dB 同身份参考视图只用于训练期排序约束。

结果是“有效改进，但暂不晋级”：

- 3/3 seeds 的 10 dB、50 dB 弱分量召回均同方向改善；
- 九档平均 FPR 增量为 `+0.0366`，满足预设 `基线+0.05` 护栏；
- 宏 AUPR 与宏 F1 均小幅改善，说明不是单纯降阈值；
- 联合已通过的参数校准器后，端到端 miss-penalty NMAE 从 `0.52816` 降至 `0.51443`，相对改善 `2.60%`，且 3/3 seeds 同方向；
- 但是 10 dB 召回均值仅 `0.45238`，未达到冻结晋级门槛 `0.8`，因此不能替换现有冻结主线，也不能申请 locked test。

## 2. 审计与实验协议

- 基座：`mlcnn_iq_grouped_task_routed` 的 parameter checkpoint；分类骨干、参数骨干和 SawFM 周期专家均冻结。
- 身份配对：相同 `sample_index` / `receiver_noise_seed` 的 30 dB 与弱功率视图配对。
- 弱分量定义：10/15 dB 削弱扫频组 SawFM、VFT、ISRJ；45/50 dB 削弱定频组 CCEL、FMB、MFT。
- 角色：沿用原全量试验的 train / early / calibration / tune / confirm 索引和哈希。
- train：残差头优化；early：checkpoint；calibration：阈值拟合；tune：配方和 beta 选择；confirm：一次性开发确认。
- locked test：未使用。

## 3. 改进结构

残差头包含三层 I/Q grouped convolution、显式 I/Q 融合、全局池化和小型 MLP。输入为当前 I/Q 特征与冻结基座 logit，输出为六类有界修正：

`logit_corrected = logit_frozen + 2 * tanh(delta_raw)`

末层零初始化，训练第 0 步严格等价于原基线。损失为：

1. 全标签 BCE，保持原六标签任务；
2. 真实弱分量正例损失；
3. 与同身份 30 dB 参考 logit 的有界排序损失；
4. 基于冻结基座高分负例的 hard-negative 约束；
5. 残差平方惩罚，限制整体漂移。

小范围 tune 筛选比较 guarded、balanced、recall 三组权重。balanced 在 seed 2041 的 tune 角色上取得最佳 10 dB 与整体弱召回，因此在正式 3-seed 阶段完全冻结。

## 4. 3-seed 分类结果

| 指标 | 冻结基线 | 可见性残差 | 变化 |
|---|---:|---:|---:|
| 10 dB 弱分量召回 | 0.16548 ± 0.02261 | 0.45238 ± 0.08172 | +0.28690 |
| 50 dB 弱分量召回 | 0.68829 ± 0.06648 | 0.81825 ± 0.01322 | +0.12996 |
| 九档平均弱分量召回 | 0.76409 ± 0.01187 | 0.86989 ± 0.02034 | +0.10580 |
| 九档平均 FPR | 0.08320 ± 0.01135 | 0.11980 ± 0.02286 | +0.03660 |
| combined macro-AUPR | 0.96926 ± 0.00165 | 0.97237 ± 0.00139 | +0.00311 |
| combined macro-F1 | 0.90997 ± 0.00225 | 0.91541 ± 0.00515 | +0.00544 |

10 dB 和 50 dB 召回均为 3/3 seeds 同方向改善，FPR 护栏通过。10 dB 仍未达到 `0.8`，promotion 判定为失败。

## 5. 端到端参数指标

参数输出使用上一轮已经验证的 visibility Ridge 校准器，分类门控改为本轮 residual 预测标签。参数校准器仍只在 early 拟合并在 tune 选择 alpha/gamma。

| 指标 | 参数校准 + 原门控 | 参数校准 + 残差门控 | 变化 |
|---|---:|---:|---:|
| Oracle parameter NMAE | 0.48919 | 0.48919 | 0（参数预测冻结） |
| Predicted-gate parameter NMAE | 0.48409 | 0.48700 | +0.00292 |
| Miss-penalty parameter NMAE | 0.52816 | 0.51443 | -0.01372（-2.60%） |
| 10 dB recall | 0.16528 | 0.45238 | +0.28710 |
| 50 dB recall | 0.68829 | 0.81825 | +0.12996 |
| 九档 FPR | 0.08325 | 0.11980 | +0.03655 |

miss-penalty NMAE 为 3/3 seeds 同方向下降。这说明新增真阳性门控带来的漏检代价下降超过新增假阳性的代价，但 2.60% 改善仍不足以支持复杂度升级。

## 6. 成果与失败原因

### 已验证成果

1. 学习型当前视图残差相对冻结基线形成了真实的可分性改善证据：宏 AUPR 上升且 3/3 seeds 的极端功率召回同方向改善，因此不只是移动阈值。
2. 30 dB 同身份参考只作为训练教师，推理无需功率标签或参考波形，满足可部署性。
3. 参数分支不重训也能因门控改善而降低端到端 miss-penalty NMAE。

但它没有超过上一轮纯后处理的 `joint_guarded` 强基线：上一轮 10 dB recall 约为 `0.564`、miss-penalty NMAE 约为 `0.5133`，本轮分别为 `0.452` 和 `0.5144`。因此本轮只能证明“学习残差相对原始冻结门控有效”，不能宣称优于现有最佳受约束方案。

### 未达标原因

1. 10 dB 是严重可辨识性边界，而非普通阈值问题。弱扫频组的物理能量被定频组压制，当前归一化后的短时局部模式仍与负例高度重叠。
2. 残差幅度很快接近边界，early 弱 AUPR 在第 3--8 轮几乎平台，说明继续增加 epoch 不会自然达到 0.8。
3. 当前单个共享残差头同时服务低功率扫频弱分量和高功率定频弱分量，两侧物理机制不同，存在容量与梯度竞争。
4. 训练只显式使用 10/15/45/50 dB，虽然中间档泛化良好，但最极端 10 dB 仍需要更有针对性的时频/周期证据。

## 7. 下一步建议与决策门

暂不把残差头接回冻结主模型。下一轮只做一个可证伪改进：将共享残差拆成两侧专家，但仍不输入真实功率值。

- side router 输入当前视图的可见性统计，软选择 fixed-weak 与 swept-weak 两个专家；
- swept-weak 专家加入 STFT / 瞬时频率序列，重点恢复 SawFM、VFT、ISRJ；
- fixed-weak 专家加入多尺度自相关与延迟峰序列，重点恢复 CCEL、FMB、MFT；
- 保留当前有界残差、hard-negative、identity-paired 排序和全部角色规则；
- 新门槛不变：10/50 dB recall 均值均不低于 0.8，3/3 seeds 同方向，FPR 不超过冻结基线 +0.05，且 miss-penalty NMAE 至少相对降低 5%。

若两侧专家仍无法使 10 dB recall 达到 0.8，应把 10 dB 标记为当前传感配置下的可识别性边界，停止继续堆叠分类头，转向数据采样或前端观测设计。

## 8. 产物

- 实现：`cj0724/models.py`、`cj0724/losses.py`
- 训练脚本：`cj0724/weak_visibility_residual_experiment.py`
- 联合评估：`cj0724/weak_visibility_joint_evaluation.py`
- 单元测试：`tests/test_weak_visibility_residual.py`
- 小范围筛选：`outputs/weak_visibility_residual_screen_20260725`
- 3-seed 正式结果：`outputs/weak_visibility_residual_full_20260725`
- 端到端联合结果：`outputs/weak_visibility_joint_eval_20260725/results.json`
