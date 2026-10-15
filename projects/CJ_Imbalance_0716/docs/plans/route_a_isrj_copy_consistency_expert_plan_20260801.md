# Route A ISRJ Copy-Consistency 周期专家预注册方案

## 结构假设

310 维周期特征稳定检测周期存在，却缺少区分“截取—重发复波形副本”与“FMB/VFT 周期调制”的信息。新增 copy-consistency 特征，在候选重复时延处保留跨窗复相关、相位一致性、包络相关、相位对齐残差和谐波时延证据。

## 固定结构

- 候选时延：归一化复信号在 40—1200 点范围内包络 ACF 最大峰。
- 将有效重叠区分为 8 个窗口，输出复相关幅度统计、跨窗相位一致性、相邻窗相位增量一致性、包络相关统计、相位对齐复制残差统计。
- 输出候选时延、全局复相关、2 倍/3 倍谐波复相关及比值。
- 新特征与原 310 维特征拼接；不删除原特征。
- HGB 容量与原专家完全相同，不搜索模型参数。
- A0：原 310D aligned HGB；A1：310D + copy-consistency aligned HGB；S1：对 block-shuffled 信号提取同样的增强特征并训练同容量 HGB。

## 执行顺序与门禁

先用 seed 2039 做 smoke，只验证有限值、形状、训练和评估链路，不作方法结论。随后五种子组件门禁：

1. A1 early matched AUROC 相对 A0 平均提升至少 0.010；
2. A1 early matched AUPR 相对 A0 非负；
3. A1 early matched AUROC 相对 S1 至少高 0.020；
4. 至少四个种子 AUROC 相对 A0 为正；
5. 至少四个种子高于 S1；
6. validation-early AUROC 平均落差小于 0.15；
7. 增强特征全部有限且维度固定；
8. aligned/shuffled HGB 容量一致；
9. test-v3 边界完整。

组件通过后才进入冻结融合验证；失败则保留现有 B2，停止继续扩展该专家。

