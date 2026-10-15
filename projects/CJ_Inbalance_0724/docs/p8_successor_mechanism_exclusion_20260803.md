# P8c 后继机制：设计阶段排除记录

## 输入证据

P8c 的三种子开发已冻结为失败：训练期 `logit=0` 上界与 calibration 得到的 VFT 概率 `0.05` 决策点不一致，且困难负样本分离度三种子均下降。这个发现可以解释 P8c，不能授权在同一机制上搜索新的 cap、系数或 margin。

代码审查还确认：`cj0724/losses.py::TargetHardNegativeLoss` 已经实现 VFT 目标的高分负样本 OHEM，并可加入正样本高于高分负样本的 pairwise ranking margin。它是先前的困难负样本/关系损失路线，不是一个未测试的新机制。组合不相交 P5d 的冻结停止规则也禁止在该证据角色上再加入 relation loss、margin 或阈值拟合来补救。

## 排除的伪后继

以下方案均不进入新的数据生成或开发训练：

1. 将 P8c cap 从 0 改为 `logit(0.05)`，或搜索 cap/系数；这是基于 P8c 已消费开发结果的超参数救援，不是独立机制。
2. 将目标困难负样本替换为 batch top-k 后再加 pairwise margin；这复用既有 `TargetHardNegativeLoss` 的 OHEM/ranking 机制。
3. 用 development FPR 重新选阈值；这把开发角色用于阈值调优，既不改善 AUPR，也破坏 P8c 的证据边界。
4. 在 P5d 的组合不相交角色上加入 margin 或 relation loss；这违反 P5d 已冻结的停止规则。

## 对后续工作的约束

多轴长尾的下一轮必须改变的不是“正例响应强度”或“同一 logit 的局部惩罚”，而是可验证的上游信息或任务结构，并先完成 oracle/物理充分性检查。若 oracle 也不能区分 VFT 正例与 `SinFM+MFT+ISRJ` 困难负例，则停止更深分类器和新损失搜索，转向前端可观测性或数据生成设计。该筛查应使用全新、train-only、身份隔离数据，并与 P8b/P8c 的失败身份零重叠。

这里的“物理充分性检查”不能重复 P2u/P2v 已完成的 VFT Doppler 家族：P2u 的有符号宽网格及 P2v 的参考对齐固定滞后相位增量均在新身份、不同 Doppler OOD 区间和错误参考/无时延对齐控制下失败，且标签 oracle 对 near-low / near-high 仍不可充分。因此，现阶段禁止再用同源 Doppler 网格、相位增量、同一候选专家的更深 router 或其损失变体作为 P8 后继。允许的新路线必须引入独立的可观测量（例如真实可用的成分隔离观测、双斜率参考或可验证时延观测），并先在新的 train-only 身份上证明 oracle 充分性。

本记录不宣称新的方法已通过；它仅防止把已失败或已覆盖的损失函数变体包装成创新点。
