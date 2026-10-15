# Route A：VFT/ISRJ 浅层 STFT 拓扑专家结果（development-v3）

## 1. 结论

3,681参数的浅层2D STFT专家没有形成可用局部证据。VFT aligned early AUROC为 `0.52610`，相对严格 final-temporal为 `-0.05006`；ISRJ为 `0.50653`，相对 final-temporal为 `-0.08609`。VFT对时间打乱仅提高 `+0.01566`，未到 `+0.020`；ISRJ对打乱为 `+0.00539` 且CI跨零。

两个标签均未通过任何主要性能门槛。按预注册，停止“加深/扩宽STFT-CNN”方向。

## 2. 结果

| 标签 | aligned AUROC | 95% CI | AUPR | 相对final-temporal | 相对time-shuffled | val→early gap |
|---|---:|---:|---:|---:|---:|---:|
| VFT | 0.52610 | [0.51184,0.53840] | 0.20559 | **-0.05006** | +0.01566 | 0.04345 |
| ISRJ | 0.50653 | [0.48702,0.52603] | 0.19351 | **-0.08609** | +0.00539 | 0.06547 |

VFT只有1/5种子优于final-temporal、3/5优于打乱；ISRJ为0/5和3/5。参数量与预算公平、泛化gap和测试边界审计通过，但不能覆盖可辨识性失败。

## 3. 原因分析

1. **局部平移不变偏置不适合当前VFT证据。** 既有固定VFT-STFT线性特征达到 `0.61658`，而CNN只有 `0.52610`。线性特征保留绝对频率位置、质心、扩展度、峰值轨迹的全局统计；卷积加全局池化更偏向“是否出现某种局部纹理”，弱化了绝对位置与完整频率轨迹。
2. **严格匹配后样本量与小validation使CNN选择不稳定。** VFT最佳epoch为2、17、2、11、2；ISRJ为16、3、10、2、3。不同种子的validation最优点变化明显，说明即使只有3,681参数，非线性模型仍容易学习角色特定纹理。
3. **ISRJ证据不在普通幅度STFT局部纹理中。** aligned接近随机且不稳定优于打乱，表明周期复位可能需要相位连续性、循环平稳性或跨较长时间尺度的结构，而不是32×32功率谱图。
4. **时间邻接不是充分条件。** VFT aligned相对打乱有小幅正增益，说明时间顺序含信息，但不足以超过冻结final-temporal，更不能达到绝对门槛。

## 4. 证据边界与下一步

- 排除：继续加深、加宽或训练更大STFT-CNN；这会在缺乏可辨识性证据时扩大容量。
- VFT保留的最佳物理证据仍是固定STFT线性probe `0.61658`，但未过0.620门槛和融合门槛。
- ISRJ下一步必须更换可证伪的物理表征假设，优先验证相位循环平稳/长周期自相关，而不是普通功率STFT。
- 任何新表征仍需严格匹配、时间/相位破坏对照、五种子和gap门槛。

## 5. 产物

- 预注册：`docs/plans/route_a_vft_isrj_tiny_stft_topology_expert_v3_plan_20260801.md`
- 代码：`experiments/run_route_a_vft_isrj_tiny_stft_topology_expert_v3.py`
- 汇总：`outputs/tapc/route_a_vft_isrj_tiny_stft_topology_expert_v3/route_a_vft_isrj_tiny_stft_topology_expert_v3_summary.json`
- 汇总 SHA-256：`97C03127335D6C3B49CE1A7769261BAC93B64787CA863229B8CE61E63668CCB3`
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`

