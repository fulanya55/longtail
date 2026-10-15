# Route A IP-CVE P0：冻结 B0 可观测性结果与停止决定

日期：2026-08-07  
状态：`failed_closed`  
证据角色：全新 identity-isolated development-only；五个正式 B0 checkpoint；无候选 backbone 训练、无校准、无阈值、无测试集。

## 1. 决策

**停止当前 IP-CVE，并禁止进入 P1 低秩 adapter、有界 residual、校准或阈值阶段。**

冻结 B0 的 matched 三元组边际信息虽然优于 unmatched、component-permuted 和 time-permuted 的部分对照，但没有战胜等维 raw-IQ random-projection 控制，且 target binding、逐标签增益与三状态顺序均未通过硬门。因此不能声称 B0 中存在可安全接回的统一三目标 counterfactual visibility evidence。

## 2. 固定协议

- 数据 SHA256：`66d7158611e92bc952cbbbd0a9243514d93d101c1297f6841911392c54564fb3`。
- fit/early/evaluation 三元组：1,440/480/480。
- B0 seeds：2039--2043，全部冻结。
- 候选 source：global、temporal-bin8、stage3-magnitude-bin8、stage4-magnitude-bin8；全部固定投影到 64 维。
- readout：`StandardScaler + LogisticRegression(C=1.0)`，仅训练 target binding diagnostic。
- 控制：unmatched background、cyclic component binding、time-permuted input、raw-IQ equal-dimension random projection。
- early 只选择 source；evaluation 只评价已选 `global_128` 及控制。

## 3. Early source selection

| Source | true weak AUROC | binding accuracy | true 相对 strongest control |
|---|---:|---:|---:|
| global_128 | 0.599786 | 0.677500 | -0.005695 |
| stage3_mag_bin8 | 0.579417 | 0.870000 | -0.018531 |
| stage4_mag_bin8 | 0.579805 | 0.837917 | -0.019510 |
| temporal_bin8 | 0.581427 | 0.793333 | -0.029911 |

按预注册选择 `global_128`。stage3/stage4 的高 binding accuracy 没有转化为单视图 weak-vs-absent 排序，说明 pair residual 能辨认“改变的是哪个目标”，但不等价于可靠的弱目标存在证据。

## 4. Evaluation 主结果

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| matched true 平均 weak AUROC | 0.550727 | — | 画像 |
| 相对 strongest control | -0.007937 | >= +0.030 | fail |
| 五 seed 95% CI | [-0.014656, -0.002349] | 下界 >0 | fail |
| 正增益 seed | 0/5 | >=4/5 | fail |
| binding accuracy | 0.584167 | >=0.60 | fail |
| binding 相对 strongest control | -0.367917 | CI 下界 >0 | fail |
| evaluation-to-early gap | 0.049060 | <0.15 | pass |
| candidate-induced non-target drift | 0 | =0 | pass |

五个 B0 seed 的 weak AUROC 差分别为 `-0.000872/-0.002187/-0.005625/-0.010326/-0.020677`，不存在 seed 筛选空间。

## 5. 对照绝对结果

| Arm | mean weak AUROC | binding accuracy | mean order fraction |
|---|---:|---:|---:|
| matched_true | 0.550727 | 0.584167 | 0.696250 |
| unmatched_background | 0.540258 | 0.477500 | 0.458750 |
| component_permuted | 0.487323 | 0.225417 | 0.307083 |
| time_permuted | 0.511813 | 0.395000 | 0.361667 |
| raw_iq_random_projection | 0.558620 | 0.952083 | 0.666667 |

matched pairing和真实时间顺序确实提供了一部分信息，但它没有超出原始 I/Q 的等维控制。P0 需要证明的是 B0 特征带来可部署的新增边际证据，而不是仅证明数据中的目标分量可以被 pair difference 看见，因此总体门失败。

## 6. 逐标签结果

| 标签 | matched weak AUROC | 相对 strongest control | 95% CI | 正 seed | order delta | 决策 |
|---|---:|---:|---:|---:|---:|---|
| MFT | 0.592742 | +0.011047 | [+0.000203,+0.021891] | 3/5 | -0.1300 | fail |
| VFT | 0.512297 | -0.037250 | [-0.043023,-0.031477] | 0/5 | -0.1375 | fail |
| ISRJ | 0.547141 | -0.101453 | [-0.103555,-0.098828] | 0/5 | -0.3250 | fail |

MFT 的小排序方向低于 +0.020 标签门、只有 3/5 seed，并且三状态顺序显著差于控制，不能晋级为窄标签候选。VFT/ISRJ 明确失败。

## 7. 根因更新

1. B0 global 表征相对 unmatched/time-permuted 有信息，但不足以形成超过 raw-IQ 的目标边际读出。
2. stage3/stage4 能高精度区分 residual 属于哪个目标，却不能稳定对单视图判断 weak presence；“target identity”与“weak existence”仍是两个问题。
3. raw random projection 的 binding accuracy 达 0.9521，说明三元组 intervention 本身可识别；但其标签行为高度异质：MFT/VFT/ISRJ weak AUROC 约为 0.5461/0.4812/0.6486，三状态 order 约为 0.9938/0.0063/1.0000。统一线性方向不是答案。
4. 当前失败不是阈值或分类头问题，因为 P0 没有校准/阈值，且 AUROC/target binding 已失败。
5. 继续训练 P1 会再次接近 H0f 的“pair difference→single-view bounded residual”链路，缺少科学准入。

## 8. 后续唯一允许方向

关闭当前单脉冲 IP-CVE。下一阶段只能审计并选择具有实质新增观测的信息协议，例如多脉冲、跨 PRI 或更长时间孔径；必须先排除与既有 H2、H0u/H1u/H2v、VFT staggered-PRI 和 ISRJ periodicity 路线的重叠。若没有非重叠且可执行的观测协议，应将结论收缩为“当前单脉冲 synthetic protocol 的成分归因边界”，而不是增加 adapter/attention/MLP。

机器结果：`outputs/route_a_ipcve/p0_frozen_observability/p0_summary.json`。  
receipt：`outputs/route_a_ipcve/p0_frozen_observability/run_receipt.json`。  
`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
