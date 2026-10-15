# 纯监督复合干扰联合参数感知实验结果

日期：2026-07-23  
环境：`D:\anaconda3\envs\PY10\python.exe`，NVIDIA GeForce RTX 5080  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat`  
边界：development roles only；所有 receipt 均为 `locked_test_used=false`

## 1. 实验目的

停止半监督路线后，验证纯监督类别条件参数回归能否：

1. 在不明显损害六类多标签识别的前提下输出复合干扰物理参数；
2. 通过参数梯度进入共享骨干，提高参数可估性；
3. 判断参数监督是否能稳定反哺分类。

每类使用 JNR 与一个类别特有参数，共 12 个参数目标。参数损失只在 `label=1 AND parameter_mask=1` 的位置计算。

## 2. 消融定义

| 变体 | 说明 |
|---|---|
| C0 | 纯分类，无参数头 |
| C1 | 参数头正常训练，但参数分支输入使用 detached backbone feature |
| C2 | 参数损失以固定权重 0.1 回传共享骨干 |
| C3 | 参数损失在前 20% 训练关闭，20%–40% 线性升至 0.1 |

模型只按 early-stop macro-AUPR 选点，参数误差不参与 checkpoint 选择。

## 3. 执行范围

1. GPU smoke：512 optimization、128 各开发角色、2 epochs、seed 2041，C0-C3 全部完成；
2. S1 筛选：4096 optimization、512 各开发角色、20 epochs、seeds 2041/2042/2043，C0-C3 共 12 组；
3. 全量确认：17,640 optimization、2,520 各开发角色、40 epochs；C1/C2 完成三个种子，C0 完成 seed 2041 sanity check，C3 完成 seed 2041 机制确认。

所有受限阶段均使用组合分层抽样，并记录每个角色的样本 SHA256。C1/C2 的逐 seed split SHA256 完全相同。

## 4. S1 三种子筛选

| 变体 | macro-AUPR | macro-F1 | Hamming | oracle 参数 NMAE | 参数覆盖率 |
|---|---:|---:|---:|---:|---:|
| C0 | 0.92024 ± 0.00486 | 0.85017 ± 0.01425 | 0.15224 ± 0.00890 | — | — |
| C1 | 0.92024 ± 0.00486 | 0.85017 ± 0.01425 | 0.15224 ± 0.00890 | 0.66035 ± 0.01518 | 0.85883 ± 0.03302 |
| C2 | **0.92125 ± 0.00493** | 0.85067 ± 0.01250 | 0.15202 ± 0.00818 | 0.63830 ± 0.01307 | 0.85711 ± 0.03284 |
| C3 | 0.92097 ± 0.00492 | **0.85212 ± 0.01420** | **0.15061 ± 0.00890** | **0.63718 ± 0.01438** | **0.85926 ± 0.03362** |

C2/C3 相对 C0 的 macro-AUPR 在 3/3 seeds 为正，均通过分类非劣门槛。C1 与 C0 的分类结果逐 seed 完全相同，证明隔离对照成立。

## 5. 全量三种子 C1/C2 确认

| 变体 | macro-AUPR | macro-F1 | Hamming | oracle 参数 NMAE | 参数覆盖率 |
|---|---:|---:|---:|---:|---:|
| C1 | 0.97444 ± 0.00142 | 0.92875 ± 0.00259 | 0.07354 ± 0.00266 | 0.61669 ± 0.01445 | 0.93129 ± 0.00559 |
| C2 | **0.97464 ± 0.00107** | **0.92893 ± 0.00283** | 0.07368 ± 0.00360 | **0.54183 ± 0.01377** | **0.93281 ± 0.00161** |

逐 seed 的 `C2-C1`：

| seed | Δmacro-AUPR | Δmacro-F1 | ΔHamming | 参数 NMAE 降低 | Δ参数覆盖率 |
|---:|---:|---:|---:|---:|---:|
| 2041 | +0.00072 | -0.00047 | +0.00152 | 0.07299 | +0.00182 |
| 2042 | -0.00054 | -0.00069 | 0.00000 | 0.07363 | -0.00482 |
| 2043 | +0.00042 | +0.00172 | -0.00112 | 0.07795 | +0.00755 |
| 均值 | +0.00020 | +0.00019 | +0.00013 | **0.07485** | +0.00152 |

参数 NMAE 从 0.61669 降到 0.54183，相对改善约 12.1%，且 3/3 seeds 同方向。分类指标只形成非劣证据：macro-AUPR 为 2/3 seeds 提高，平均增益约 0.00020；macro-F1 与 Hamming 的变化也接近零。因此不能声称参数监督显著提高识别，只能声称它在基本保持识别性能时显著改善参数估计。

## 6. 分参数结果

三种子平均有 11/12 个参数的 NMAE 改善。改善最大的目标：

| 参数 | C1 MAE | C2 MAE | NMAE 降低 |
|---|---:|---:|---:|
| FMB interference bandwidth | 458,791 Hz | 316,554 Hz | 0.19610 |
| ISRJ sample width | 3.492e-6 s | 2.700e-6 s | 0.18312 |
| SinFM frequency deviation | 257,818 Hz | 172,411 Hz | 0.14781 |
| SinFM JNR | 2.698 dB | 2.288 dB | 0.07061 |
| SawFM JNR | 2.801 dB | 2.392 dB | 0.07053 |

六类 JNR MAE 均改善。两个未充分解决的参数是：

- MFT `num_targets`：1.4588 → 1.4517，几乎没有改善。数据审计显示它只有 3–8 六个近似均衡离散值，应改为 ordinal/classification head，而不是连续 Smooth L1；
- VFT `Doppler`：1482.55 → 1483.39 Hz，轻微变差。该参数在 1000.77–6999.32 Hz 连续分布，当前全局池化时域骨干可能缺少直接频移表征，应增加频谱/时频参数支路。

## 7. C3 判断

全量 seed 2041 中，C3 的 macro-F1/Hamming 略优于 C2，但参数 NMAE 与 C2 基本相同，参数覆盖率更低，macro-AUPR 增益也更小。结合 S1 结果，不继续扩展 C3；C2 作为主模型，保持实现和实验链路简单。

## 8. 最终结论与下一步

当前可成立的结论：

> 类别条件的纯监督参数联合训练能够在基本保持六类复合干扰识别性能的同时，稳定改善物理参数估计；主要收益体现在参数感知，而不是分类提升。

下一轮不应继续扫统一参数权重。优先进行参数类型适配：

1. MFT `num_targets` 改为六级 ordinal/classification head；
2. VFT `Doppler` 增加 FFT magnitude 或轻量时频支路，并与共享语义特征融合；
3. 其余连续参数保留 normalized Smooth L1；
4. 新实验仍以 C1 detached baseline 和 C2 shared-gradient baseline 为对照，只复验 MFT/VFT 两个失败参数；
5. 在参数类型适配完成并通过开发集三种子后，再冻结模型、参数头、温度和阈值，申请一次 locked-test 评估。

## 9. 产物与验证

- 方案：`docs/implementation/supervised_joint_parameter_experiment_plan_20260723.md`
- runner：`experiments/run_supervised_joint_parameter.py`
- 模型：`cjlt/models/supervised_joint.py`
- 训练与指标：`cjlt/engine/supervised_joint_trainer.py`
- S1 汇总：`outputs/student_b/supervised_joint_parameter_screen_20260723/supervised_joint_parameter_summary.json`
- 全量 seed 2041：`outputs/student_b/supervised_joint_parameter_confirm_seed2041/`
- 全量 seeds 2042/2043：`outputs/student_b/supervised_joint_parameter_confirm_final/`
- C0 全量 sanity：`outputs/student_b/supervised_joint_parameter/C0/seed_2041/`
- 全量测试：`127 passed`，仅有 36 个既有 AMP 弃用警告。

