# P4g SawFM 周期目标平衡损失失败审计

## 研究问题

在不改变网络结构和推理成本的前提下，把 SawFM sweep-period 的有效回归项加权为 2×，能否消除 P4c/P4d 中已复现的周期负迁移，同时保持其余 11 参数和分类性能？

预注册文件：`configs/p4g_sawfm_period_target_balanced_loss_preregister_20260801.json`

预注册 SHA256：`b59bee61fe9f855ce29fbe4e685a5bf9a38b8c7a3119c3445d8fd159dc4a22f9`

## 实验协议

- C0：`mlcnn_iq_grouped_task_routed`，12 个有效目标权重均为 1；
- C1：`mlcnn_iq_grouped_task_routed_saw_loss2`，仅 SawFM sweep-period 权重为 2；
- 网络、参数量、推理图、训练轮数和 checkpoint objective 完全相同；
- 加权损失按有效目标权重总和归一化；
- 数据生成种子 710000000，JNR 10–30 dB，63 组合 × 100 身份；
- 每组合 train/early/calibration/tune/confirm = 75/10/5/5/5；
- 模型种子 6041、6042、6043；
- 20 epochs，AdamW，cosine LR 1e-4，weight decay 1e-4，warmup 2，batch 63，dropout 0.2，参数损失系数 0.1。

身份审计：6300/6300 唯一；`receiver_noise_seed` 范围 908677743–908684042；与 root 和 P4b–P4f 零重叠。confirm 含 315 个身份并覆盖全部 63 组合。12 个目标均有限、可变，mask 与标签一致。

## 三种子结果

| seed | C0 full-12 NMAE | C1 full-12 NMAE | full-12 Δ，越低越好 | Saw 周期 C0 | Saw 周期 C1 | Saw 改善 | Pearson Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6041 | 0.652475 | 0.655626 | +0.003151 | 0.839962 | 0.836409 | +0.003553 | +0.039908 |
| 6042 | 0.648636 | 0.650733 | +0.002097 | 0.834548 | 0.832441 | +0.002108 | +0.014020 |
| 6043 | 0.650339 | 0.654782 | +0.004443 | 0.840112 | 0.838306 | +0.001805 | -0.000028 |
| 均值 | — | — | +0.003230 | — | — | +0.002489 | +0.017966 |

分类 Macro-AUPR 的平均变化约 +0.000016，未发生实质分类退化。两种模型参数量均为 12,890,786，推理图完全相同。

## 预注册判定

失败门槛：

- 每种子 SawFM 周期改善至少 0.015：失败；
- 三种子均值改善至少 0.025：失败；
- 每种子周期 Pearson 不下降：失败；
- 每种子 full-12 NMAE 恶化不超过 0.003：失败；
- full-12 NMAE 均值不得恶化：失败。

通过项目：分类门槛、参数量/推理图一致性、其余 11 参数没有出现 3/3 且绝对恶化超过 0.02 的单目标系统性失败。

最终判定：`promotion_passed=false`。

## 失败根因

2× 权重确实使前两个种子的周期 Pearson 提高，说明优化器对周期目标的响应增强了；但 NMAE 改善只有约 0.0025，说明响应增强没有转化为有效的绝对数值可分性。同时 full-12 NMAE 平均恶化约 0.00323，表明共享 IQ 参数路径发生了容量再分配和轻微负迁移。

所以根因不是“周期损失权重还不够大”，而是当前 IQ 表征对周期信息的物理充分性不足，并与多目标回归存在冲突。继续扩大权重只会在同一数据上进行事后搜索，不能构成可靠创新。

## 实现错误边界

第一次运行 `tuned_multitask_20260801_072741` 在掩码前对含 inactive NaN 的完整目标张量计算 SmoothL1，可能污染反向梯度，因此已中止并标记为无效，不用于任何方法比较或模型选择。

修复后，损失仅对 `valid AND finite` 元素计算，并增加回归测试，要求 inactive NaN 存在时预测梯度仍全部有限。有效证据仅来自 `tuned_multitask_20260801_073107`。

## 停止项与保留成果

停止：

- SawFM 周期 3×/4× 或连续权重搜索；
- detached learned SawFM fallback；
- 在没有新物理充分性证据时构造 SawFM 深度专家。

保留：

- P4c/P4d 的总体异构路由收益；
- SawFM 周期是稳定负迁移边界的逐目标审计结果；
- NaN-safe masked parameter loss 的实现和回归测试。

下一步冻结 P4c 候选，转向严格等算力五种子统计、九功率/弱分量、参数 OOD、组合不相交和多轴长尾证明。

证据边界：`test_generated=false`，`test_opened=false`，`locked_test_used=false`。
