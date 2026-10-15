# Route A 严格双向组合匹配协议纠正 development-v3

## 问题

原 `_matched_view` 只过滤负样本，却保留没有任何同组合负样本的弱正。development-v3 长尾 fit/validation 中，MFT 仅约 57%--64%、VFT 仅约 50%--60% 的弱正真正有同角色匹配负样本；early 因样本更大几乎全部可匹配。这会让探针利用“只在正类出现的组合”捷径，并人为放大 validation→early 落差。

## 纠正

严格视图仅保留“其他五标签组合同时存在弱正和目标缺失负样本”的组合，正负两侧均取交集。此前所有 layerwise、raw-physics、VFT coarse/Ridge 可分性结果降级为探索性证据，不用于模型晋级。用新输出目录重新执行：

1. 四层冻结可分性五种子；
2. MFT/VFT/ISRJ 固定原始 I/Q 物理摘要五种子；
3. 仅在严格协议结果支持时继续 VFT 稳定化或专家接入。

门槛保持原预注册值，不因纠错后的结果调整。test-v3 继续封存。
