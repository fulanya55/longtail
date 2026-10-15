# Route A 创新 Goal 最终审计与停止决定

日期：2026-08-01

## 最终决定

当前路线达到“协议内有效、统计为正、优于所比较基线”，但没有达到“效应量跨
fresh development 稳定”的预注册要求。因此：

**可靠通用创新点尚未被证明，当前 Goal 以预注册 stop 结论终止。**

不再生成新的 development 数据，不进入架构/分布鲁棒性或外部一次性测试，避免
顺序试验和选择性报告。

## 四个最终问题

### 1. 性能提升是否真实、稳定并具有统计意义？

- macro-AUPR 和 macro-F1：真实且跨 v3/v4/v5/v6 稳定为正；
- target weak recall：所有轮次均为正，多个 CI 排除 0；
- 但效应量分别约 `+0.03283、+0.01691、+0.02813、+0.02454`，未稳定达到固定
  `+0.025/+0.030` 门槛。

回答：**统计提升真实，但关键弱分量效应量不够稳定。**

### 2. 是否由 SawFM 特异配对方向产生？

- midpoint 只能解释部分收益；
- other-label directions 整体更差；
- 正交任务子空间控制 weak recall 为负；
- 非配对和高维随机等范数方向失稳；
- 但 shared-only 方向有害，classifier-only 可获得大部分收益。

回答：**支持严格配对和 SawFM label-deletion direction 的特异性，但不支持“纯共享
梯度冲突抑制”解释；更准确机制是 shared+classifier task-vector 协同。**

### 3. 是否优于通用梯度/多任务基线？

- strongest baseline 是 M5；
- 相对 M5，候选 weak recall `+0.00853`，95% CI `[+0.00384,+0.01408]`；
- 相对 PCGrad/MGDA/CAGrad/GradNorm 也更高；
- 方法优势 gate 通过。

回答：**在当前两任务公平协议中是，但这不足以覆盖全部文献实现和架构。**

### 4. 是否跨 SNR、长尾、骨干和外部数据泛化？

由于候选在 v6 未通过自身效果量 gate，按预注册规则不得晋级后续阶段。

回答：**没有得到证明，也不应声称已经泛化。**

## 最强允许表述

> 在保留 SawFM 共现背景的六类复合干扰识别中，C6 与 SawFM-label-deletion M5
> 之间的严格配对 task vector，经固定 alpha=1.5 外推后，在多个身份隔离 development
> 和一次独立 test-v2 上获得一致的 macro-AUPR/macro-F1 增益，并在当前两任务协议
> 中优于 M5、PCGrad、MGDA、CAGrad 和 GradNorm。其弱分量 Recall 增益为正但
> 效应量跨 fresh development 波动，因此尚不构成跨分布、跨架构的可靠通用创新。

## 禁止表述

- “提出并验证了通用梯度冲突解决方法”；
- “LSCP 获得验证”；
- “方法稳定降低 FPR”；
- “已经跨分布、跨骨干或外部数据验证”；
- “可继续使用 test-v2 调参或确认”。

## 证据边界

- 历史 locked-test 未访问；
- test-v2 已在先前一次性流程中消耗，之后未重跑；
- test-v3 始终密封；
- v4/v5/v6 均为 train-only development；
- 本 Goal 未进行任何新的 independent/external test；
- 终止原因来自预注册 gate，而非资源或实现阻塞。
