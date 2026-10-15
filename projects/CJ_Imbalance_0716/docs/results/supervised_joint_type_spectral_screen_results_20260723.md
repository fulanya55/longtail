# 有监督联合参数感知 S2：类型化 MFT 与频谱 VFT 验证结果

## 结论

本轮两个针对性假设均未通过：普通六分类头没有解决 MFT `num_targets`，全局 FFT 功率谱摘要也没有解决 VFT `Doppler`。两类输出都接近常数预测，因此失败根因已经从“损失形式或缺少频谱”进一步收窄为“当前输入到目标的可辨识特征尚未被提取，或样本参数与复合波形之间存在更基础的可辨识性/对齐问题”。按预注册门槛停止，不进入完整开发集确认，也未使用 locked test。

## 协议

- 基线：C2，共享主干、联合参数梯度。
- 候选：`C2_MFT_CE`、`C2_VFT_FFT`、`C2_TYPED_FFT`。
- 每个变体 3 seeds（2041/2042/2043）。
- 每个 seed：train 4096；early-stop/calibration/threshold-selection 各 512；20 epochs，patience 5。
- 同 seed 的四个变体具有相同 split/sample 哈希。
- locked test used：false。

## 三种子均值

| 变体 | macro-AUPR | macro-F1 | Hamming | 参数 NMAE | MFT rounded MAE | MFT rounded acc | VFT Doppler MAE/Hz |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 0.921245 | 0.850675 | 0.152018 | 0.638296 | 1.487678 | 0.168612 | 1491.040 |
| C2_MFT_CE | 0.921430 | 0.852474 | 0.150933 | 0.641154 | 1.482490 | 0.167315 | 1491.518 |
| C2_VFT_FFT | 0.921303 | 0.849554 | 0.151801 | 0.638232 | 1.490272 | 0.167315 | 1490.694 |
| C2_TYPED_FFT | 0.921510 | 0.850084 | 0.152344 | 0.641160 | 1.482490 | 0.167315 | 1494.116 |

## 相对 C2 的门槛核验

### MFT 六分类

- rounded MAE 只降低 0.00519，即 0.35%，远低于 15% 门槛；三个 seeds 的变化为 -0.00778、+0.00778、-0.01556。
- rounded accuracy 平均下降 0.00130。
- 分类 macro-AUPR 非劣（+0.000185），但总体参数 NMAE 变差 +0.00286。
- 结论：普通 CE 仅改变输出形式，没有产生有效的数量判别信息。

### VFT 全局频谱支路

- Doppler MAE 只降低 0.345 Hz，即 0.023%，远低于 10% 门槛；三个 seeds 的变化为 -21.02、+4.91、+15.07 Hz，方向不稳定。
- 分类 macro-AUPR 非劣（+0.000058），总体参数 NMAE 基本不变（-0.000064）。
- 结论：整段 FFT 功率谱没有保留足够的 VFT 时变频率轨迹。

### 组合项

- MFT 与单独 CE 相同，VFT 反而平均变差 3.08 Hz。
- 没有观察到两项弱改造组合后的协同收益。

## 失败诊断

将三 seeds 的 early-stop 预测合并后：

- MFT 真值标准差为 1.713；C2 预测标准差仅 0.159，CE 后进一步缩到 0.105；预测与真值相关系数只有 0.053。CE 模型 rounded accuracy 0.1673，低于开发样本多数类常数基线 0.1907；rounded MAE 1.4825 与中位数常数基线 1.4864 几乎相同。
- VFT 真值标准差为 1711.5 Hz；C2 预测标准差仅 106.6 Hz，FFT 支路虽增至 230.5 Hz，但与真值相关系数只有 0.0356；其 1490.7 Hz MAE 仍差于中位数常数基线 1482.1 Hz。
- 这不是训练崩溃：分类指标稳定、其他十个参数仍维持上一轮水平。失败集中在两个专项头，并呈现明显均值塌缩。

因此，当前不能再简单增加联合损失权重或扩大训练轮数。普通分类头仍依赖没有数量信息的共享嵌入；全局 FFT 又抹掉了 VFT 的时间变化过程。

## 下一步建议：先做可辨识性探针，再回接联合模型

1. **P0 参数对齐审计**：抽样核对 MAT 参数列、标签、波形生成记录与样本索引；按复合阶数、目标 JNR/功率和共现类别分层，确认参数确实对应当前复合波形中的该分量。
2. **P1 oracle-positive specialist**：关闭分类损失，只在已知 MFT/VFT 阳性且参数有效的训练样本上训练单参数专家。若专家都不能显著击败常数基线，则联合模型不可能解决，应优先处理数据可辨识性。
3. **MFT 结构化表征**：使用多尺度自相关/延迟峰序列支路，再测试 ordinal（CORAL）目标；核心是先提取可数的重复延迟结构，而不是只换 CE。
4. **VFT 时频表征**：使用短时傅里叶谱图或瞬时频率序列，保留时间轴并做轻量 2D/时序回归；不再使用整段 FFT 池化。
5. **晋级门槛**：专项专家相对常数基线 MAE 至少降低 10%、相关系数至少 0.3，并在 3/3 seeds 同方向；通过后才把该支路接回 C2，继续检查分类非劣。

## 产物

- 实现：`cjlt/models/supervised_joint.py`、`cjlt/engine/supervised_joint_trainer.py`、`experiments/run_supervised_joint_parameter.py`。
- 测试：`tests/test_supervised_joint_parameter.py`，新增专项测试后 6/6 通过。
- 方案：`docs/implementation/supervised_joint_type_spectral_validation_plan_20260723.md`。
- 结果：`outputs/student_b/supervised_joint_type_spectral_screen_20260723`。
