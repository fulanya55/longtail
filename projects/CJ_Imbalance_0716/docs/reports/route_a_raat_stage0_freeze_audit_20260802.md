# Route A RAAT 阶段 0 冻结审计

日期：2026-08-02  
结论：**通过阶段 0；仅授权进入“按契约实现并冻结代码”的步骤，未授权生成数据或训练。**

## 已核验项

| 项目 | 结果 |
|---|---|
| 历史失败机制 | 已逐项列为永久禁止：重加权、查询/任务向量、logit/width/scale 残差、STFT 拼接、pair margin/prototype、现有 unified complex 分支等 |
| 候选数 | 3 个以内；仅 RAAT 入选，VFT 盲相位回归与 ISRJ 尺度投票均明确拒绝 |
| 数学与部署输入 | 固定为 I/Q + 已知 LFM；禁用真/预测功率、参数、seed、query、logits、阈值和校准 |
| 三目标异质性 | MFT=延迟多重性、VFT=去调频滞后相干、ISRJ=分段延迟碎片；各自独立头和 H0 |
| 因果控制 | 同形状反斜率错误参考；强制 map 差异、初始化/容量/batch/步数/标签一致审计 |
| 网络和训练 | 固定 7,425 参数、BCE、AdamW、30 epoch、5 seeds、无 early stop/无 checkpoint 选择 |
| 身份 | 预留 `978677743--978682782`（5040）；材料化前必须再与所有历史 train、H0e--H0h 及 test-v3 记录做交集审计 |
| 门与停机 | 单目标 H0：`+0.015`、paired CI 下界 `>0`、4/5 正向、每种子 strong 不低于 `-0.010`；少于两个标签通过即不进融合 |
| 测试边界 | `locked_test_used=false`、`test_v2_used=false`、`test_v3_opened=false`、`test_v3_used=false` |
| 阶段 0 文档完整性 | 三份冻结文件 SHA-256 与 manifest 一致 |

## 冻结文件

| 文件 | SHA-256 |
|---|---|
| `docs/reports/route_a_raat_stage0_literature_overlap_audit_20260802.md` | `1e1bbe8b59116637eff55cc2b2cc3ea0ec81d3799e5fb3a872c4335d0819372d` |
| `docs/plans/route_a_raat_h0_preregistration_20260802.md` | `0959444952a3d6cfed06c241339b5fb9df8a210d2aa88f6768a063b759df70be` |
| `docs/plans/route_a_raat_h0_implementation_contract_20260802.md` | `cd2c4920016039e2555b2ee9aa754f7b72974c6379f9ca35cefeef96a9fa88e5` |
| `outputs/tapc/route_a_raat_stage0_freeze/stage0_preimplementation_manifest.json` | 见本次下方外部哈希核验 |

## 还未完成，且不得暗中替代

- 没有 RAAT 特征实现、网络实现、单元测试、CPU smoke、正式数据、训练、预测、指标或 checkpoint。
- H0g/H0h 的 ISRJ aligned-periodicity 结果属于另一条已冻结窄机制，不能作为 RAAT 成功、控制臂、训练初始化或融合依据。
- 文献矩阵对未取得全文的方法条目只保留 `[T]`，不据此作具体网络/损失比较。

## 唯一下一步

严格实现 `route_a_raat_h0_implementation_contract_20260802.md` 中列出的五个文件，执行内存 CPU smoke 和静态禁用项检查，然后生成一个不可改写的 implementation manifest。若实现与预注册不符，应停止并报告不符项；不得改数学、网格、网络、超参或数据范围来使其运行。
