# Route-A P4b：严格匹配元数据审计与受限筛查（2026-08-28）

## 状态

**partial / data-limited，不能作为严格身份反事实证据。** 本轮尝试将 P4 的困难负样本配对升级为 target range、delay、measured power 和 target-present 条件匹配；但数据字段审计表明这些条件在当前 `closedset_9class_train.mat` 中不可观测。

## 数据可观测性审计

30 dB train 文件中：

- `receiver_noise_seed`：186,000 个样本全部唯一，无法在文件内找到同 receiver 身份的另一组合；
- `target_range_m`：100% NaN；
- `target_delay_s`：100% NaN；
- `target_measured_power_dB`：100% NaN；
- `target_present`：恒为 0；
- 可用的逐样本条件主要仍是六类标签与 component effective JNR。

因此本轮的“metadata matching”实际退化为在 target-present 常量条件下寻找标签数相同、组合不同的配对，不能宣称完成 receiver/background 匹配或去除单成分反事实生成。

## 受限结果

在与 P4 相同的 3-seed、2-epoch、IF=5 长尾清单下，结果如下：

| 变体 | 验证 Macro-AUPR | 验证 Macro-F1 | 验证组合完全匹配 |
|---|---:|---:|---:|
| F2+BCE | 0.999906±0.000149 | 0.997079 | 0.982096 |
| F2+BCE+matchedHN | **0.999919±0.000129** | **0.997130** | **0.982259** |

matched-HN 的组合率提升约 0.02 个百分点，但功率开发集没有一致收益，且与 P4 的近似配对结果基本同量级，不能证明严格身份匹配带来了额外贡献。

## 判定

该实验的主要成果是确认了**数据字段不足以支持当前严格反事实配对主张**，而不是确认了新的模型创新。P4 的有效证据仍限于“同标签数、总 JNR 或可用条件最近”的近似困难负样本；严格 receiver/background 对和单成分去除对需要重新生成带身份元数据的数据，或使用另一份明确提供这些字段的数据。

## 工件

- [身份/元数据审计](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_identity_audit_20260828\identity_metadata_audit.json)
- [复用的执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p4_matched_hard_negative.py)
- [受限筛查汇总（v2）](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4b_metadata_hn_v2_20260828\p4_matched_hn_summary.json)
- [角色清单（v2）](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4b_metadata_hn_v2_20260828\role_manifest.json)
- [终端回执（v2）](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4b_metadata_hn_v2_20260828\terminal_receipt.json)

## 后续决策

不把 P4b 晋级为 P5 的依据。下一步应先完成数据生成/字段补全审计：为每个样本保存 receiver/background identity、目标条件和成分级反事实索引；若短期无法补全，则把 P4 近似困难负样本结论限定为“部分支持”，不构造深度专家。
