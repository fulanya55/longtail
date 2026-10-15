# 路线B C2参数选择性梯度筛选结果

日期：2026-07-25  
环境：`D:\anaconda3\envs\PY10\python.exe`，NVIDIA GeForce RTX 5080  
边界：仅development roles，全部receipt均为`locked_test_used=false`

## 1. 本轮假设

C2已验证能够相对C1将参数NMAE降低约12.1%，但分类macro-AUPR仅提高约0.00020。本轮检验：分类增益不稳定是否主要来自不同参数任务对共享分类主干的梯度冲突。

实现了三个对照：

| 变体 | 定义 |
|---|---|
| C2 | 原固定权重0.1统一回传 |
| C2-GP | 12个参数逐目标计算梯度，负余弦时投影掉与分类梯度冲突的分量 |
| C2-IGP | C2-GP基础上，MFT `num_targets`和VFT `Doppler`只训练私有参数头，不更新共享主干 |

额外完成C2-SD seed 2041，单独阻断MFT/VFT，作为机制隔离。

## 2. 协议

- optimization：4096；early-stop/calibration/threshold-selection：各512；
- epochs：20；seeds：2041、2042、2043；
- 三变体逐seed使用相同split SHA256；
- 参数损失仍严格限定为`label=1 AND parameter_mask=1`；
- 参数头始终训练，选择性策略只控制参数梯度是否进入共享主干；
- 每个epoch审计一个batch的12目标梯度余弦和范数。

## 3. 三种子结果

| 变体 | macro-AUPR | macro-F1 | Hamming | 参数NMAE | 参数覆盖率 |
|---|---:|---:|---:|---:|---:|
| C2 | 0.920831 | 0.851405 | 0.152344 | **0.640061** | **0.863138** |
| C2-GP | **0.920903** | **0.851570** | **0.151259** | 0.640309 | 0.860340 |
| C2-IGP | 0.920888 | 0.851117 | 0.151801 | 0.640221 | 0.860555 |

C2-GP相对C2的逐seed macro-AUPR差值为：

- seed 2041：+0.000150；
- seed 2042：-0.000029；
- seed 2043：+0.000093；
- 平均：+0.000071。

参数NMAE变化为`+0.000248`（数值变差），且覆盖率降低约0.00280。C2-IGP的macro-AUPR平均仅提高0.000056，参数NMAE同样略差。二者均未达到进入全量确认的增益门槛。

C2-SD在seed 2041相对C2的macro-AUPR为-0.000030，参数NMAE仅改善0.000034，说明直接阻断MFT/VFT没有独立可见收益，因此没有扩到三种子。

## 4. 梯度机制结果

- C2每个epoch审计batch中，负梯度目标比例三种子平均为43.75%；
- C2-GP每个epoch平均投影5.25/12个参数目标；
- C2-IGP在10个允许迁移目标中平均投影4.35个；
- 梯度冲突是真实存在的，但冲突符号跨seed和训练阶段变化明显。

稳定性较弱的例子：

- ISRJ JNR三种子平均余弦为-0.0156，2/3 seeds为负；
- ISRJ sample width、FMB bandwidth、MFT JNR等只在1/3 seeds为负；
- SawFM JNR、VFT JNR在3/3 seeds均为正；
- MFT `num_targets`与VFT `Doppler`在本轮3/3 seeds平均余弦均为正，说明“参数难预测”等价于“梯度一定损害分类”的假设不成立。

## 5. 失败原因

1. **冲突是动态噪声，不是稳定的任务属性。** 同一参数在不同seed或epoch会改变符号，逐batch投影不能形成稳定归纳偏置。
2. **参数权重本来只有0.1。** 冲突梯度对总更新的贡献有限，投影后的模型与C2高度接近，所以最终差异只有1e-4量级。
3. **梯度手术没有增加新信息。** 它只能改变已有全局池化参数特征的优化方向，不能恢复池化前丢失的局部时频/时延证据。
4. **可辨识性与梯度一致性不是同一概念。** MFT/VFT预测失败主要来自观测和表示不足，但其平均梯度仍可能与分类同向；仅凭参数难度做静态detach缺少依据。
5. **计算代价不匹配收益。** 逐目标梯度投影使训练时间从C2约68秒/seed增加到约143秒/seed，收益不足以支持部署或全量扩展。

## 6. 决策

本轮结论为负：

> 保留梯度审计作为诊断工具，但停止将PCGrad、静态MFT/VFT detach或二者组合视为路线B主创新，不进入17,640样本全量确认。

下一轮应改变信息入口，而不是继续调整梯度组合：

1. 从共享主干池化前时序图提取标签局部参数特征；
2. 先在FMB bandwidth、ISRJ sample width、SinFM frequency deviation三个已验证改善最大的连续参数上做frozen/oracle-positive probe；
3. 必须与当前全局池化参数头和常数基线比较；
4. 只有局部probe达到相对全局头NMAE降低至少5%、3/3 seeds同方向，才接回C2；
5. MFT继续独立ACF/匹配滤波路线，VFT继续停止通用时频头扩展。

## 7. 产物

- 训练器：`cjlt/engine/supervised_joint_trainer.py`
- 参数损失拆分：`cjlt/losses/parameter.py`
- runner变体：`experiments/run_supervised_joint_parameter.py`
- 聚合器：`experiments/aggregate_supervised_joint_igp.py`
- 单元测试：`tests/test_supervised_joint_parameter.py`
- 运行目录：`outputs/student_b/supervised_joint_igp_screen_20260725`
- 汇总：`outputs/student_b/supervised_joint_igp_screen_20260725/supervised_joint_igp_summary.json`
