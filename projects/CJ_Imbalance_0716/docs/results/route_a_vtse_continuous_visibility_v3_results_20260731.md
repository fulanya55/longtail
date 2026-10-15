# Route A 连续可见度加权 development-v3 结果

## 1. 结论

本轮在不改变网络容量、RawSaw、VBF、数据角色、校准器和 FPR 约束阈值的条件下，
将 VTSE 从“弱正独立目标”改为 all-positive presence BCE，并按训练期相对功率差连续提高
低可见度正样本权重。五随机种子正式结果未通过晋级门槛，因此不晋级，也不访问 test-v3。

连续加权相对 hard-mask C3 的弱召回和弱正—负样本 AUROC 均稳定改善；但相对同容量
all-positive C4 只获得很小的阈值后弱召回增益，排序 AUROC 反而下降。这表明样本加权主要
改变工作点，并没有形成独立、可靠的物理可见度排序能力。

## 2. 固定设置

- development-v3 SHA256：`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`
- 随机种子：2039、2040、2041、2042、2043
- 可见度：`gap = P_label - max(P_other positive)`
- 连续严重度：`sigmoid((-10-gap)/2.5)`
- 正样本权重：`1 + severity`；负样本权重 1；保留逐标签 `pos_weight`
- 网络新增参数与 C3/C4 相同；不搜索中心、温度或最大倍数
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`

## 3. 五种子均值

| 方法 | macro AUPR | macro F1 | 目标弱召回 | 弱正-vs-负 AUROC | 目标强召回 | 目标 FPR |
|---|---:|---:|---:|---:|---:|---:|
| C0 F2+C6 | 0.85050 | 0.73409 | 0.29937 | 0.58902 | 0.79720 | 0.18812 |
| C3 hard-mask unified | 0.87838 | 0.76623 | 0.30896 | 0.59167 | 0.79870 | 0.19177 |
| C4 all-positive same capacity | **0.88128** | **0.77043** | 0.31192 | **0.59568** | **0.81290** | 0.19032 |
| S2 continuous visibility | 0.87993 | 0.76941 | **0.31503** | 0.59494 | 0.81022 | 0.19355 |

SawFM 继续完全继承 RawSaw：AUPR 0.93050，弱召回 0.58259。

## 4. 配对差值

S2 相对 C0：

- macro AUPR `+0.02944`
- macro F1 `+0.03532`
- 弱召回 `+0.01566`，5/5 种子为正
- 弱正—负 AUROC `+0.00592`
- 强召回 `+0.01302`
- FPR `+0.00543`

S2 相对 C4：

- macro AUPR `-0.00135`
- macro F1 `-0.00102`
- 弱召回 `+0.00310`，95% bootstrap CI `[+0.00062,+0.00548]`
- 弱正—负 AUROC `-0.00073`，95% bootstrap CI `[-0.00112,-0.00012]`
- 强召回 `-0.00269`
- FPR `+0.00323`

## 5. 门槛与失败原因

通过：优于 hard-mask、FPR 不超过 0.20、SawFM 保持、测试边界安全、相对 C0 的弱召回
5/5 种子为正。

失败：相对 C0 的弱召回未达到 `+0.030`，AUROC 未达到 `+0.015`；相对 C4 的 AUROC
没有非劣。连续权重产生的召回增益伴随更高 FPR，且没有排序增益，所以不能作为独立创新
机制证据。

依据预注册规则，停止继续搜索样本权重、中心和温度。下一步先验证冻结 temporal feature
是否能够回归物理相对功率差；若不能，则应转向同 receiver identity 的多功率配对与 stage2
小型 adapter，而不是继续改分类损失。

## 6. 产物

- 汇总：`outputs/tapc/route_a_vtse_continuous_visibility_v3/route_a_vtse_continuous_visibility_v3_summary.json`
- 每种子：模型、训练历史、校准器、阈值、预测和指标均保存在对应 `seed_*` 目录
- 实现：`experiments/run_route_a_vtse_continuous_visibility_v3.py`
- 测试：`tests/test_route_a_vtse_continuous_visibility_v3.py`

