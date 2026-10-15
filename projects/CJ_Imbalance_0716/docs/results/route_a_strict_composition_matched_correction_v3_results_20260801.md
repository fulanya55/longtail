# Route A 严格双向组合匹配协议纠正结果

## 1. 协议纠正影响

原匹配实现保留了没有同组合负样本的弱正，造成组合捷径。严格双向交集后，MFT/VFT 的 fit 与 validation 弱正只保留真正存在同组合负样本的部分；early 因样本规模较大基本全部可匹配。旧的 layerwise、raw-physics、VFT coarse/Ridge 结论均降级为探索性，不再作为晋级证据。

## 2. 严格分层审计

| 表征 | early AUROC | gap | 结论 |
|---|---:|---:|---|
| Global | 0.58482 | 0.20230 | 不通过 |
| Final-temporal | **0.60698** | 0.17321 | 最强但未通过 |
| Stage2 | 0.53053 | 0.14679 | 不通过 |
| Stage3 | 0.56989 | **0.11935** | 不通过 |

Final-temporal 逐标签：MFT `0.65217`、VFT `0.57616`、ISRJ `0.59262`。这表明 MFT 在冻结 final-temporal 中已有可用的严格匹配证据，但三标签平均和 VFT/ISRJ仍不足。

## 3. 严格原始 I/Q 物理审计

| 标签/特征 | early AUROC | 相对严格 final | gap | 正增益种子 | 结论 |
|---|---:|---:|---:|---:|---|
| MFT-ACF | 0.48160 | -0.17057 | 0.20509 | 0/5 | 排除 |
| VFT-STFT | **0.61658** | **+0.04042** | **0.10039** | 4/5 | 接近但未达0.620 |
| ISRJ-Reset | 0.49370 | -0.09891 | 0.15347 | 0/5 | 排除 |

VFT-STFT 已通过增益、种子一致性和泛化 gap 三项门槛，但绝对 AUROC 比 `0.620` 低 `0.00342`，因此不能晋级。其 CI `[0.58566,0.64625]` 也表明种子波动仍需处理。

## 4. 当前可信结论

1. 旧“匹配负样本”协议存在单向过滤缺陷，已修正并增加单元测试；
2. 冻结 F2 的 stage2/stage3 不适合作为统一三目标局部专家入口；
3. MFT 的可用证据位于 final-temporal，而固定自相关反而破坏可分性；
4. VFT 的 STFT 时频证据具有独立、跨角色可复现增益，是下一轮唯一允许继续的原始物理分支；
5. ISRJ 当前无通过门槛的冻结层或固定物理摘要，必须提出新的机制，不能沿用现有 reset 全局统计；
6. test-v3 仍未授权、未使用。

## 5. 下一步单一试验

只验证 VFT 的互补性：冻结严格 final-temporal 探针与严格 STFT 探针，使用无参数的 fit 标准化等权分数融合，并以角色内打乱 STFT 分数作机制对照。若融合绝对 AUROC达到 `0.62`、相对两单支均有稳定增益且 gap `<0.15`，才允许将 VFT-STFT作为统一模型受限残差分支。MFT 与 ISRJ不在该轮同时修改。

产物：

- `outputs/tapc/route_a_layerwise_matched_negative_separability_v3_strict/route_a_layerwise_matched_negative_separability_v3_summary.json`
- `outputs/tapc/route_a_raw_iq_physics_separability_v3_strict/route_a_raw_iq_physics_separability_v3_summary.json`
