# Route A Frequency Reconstruction V2 Manifest 可训练性补充

日期：2026-08-08  
触发阶段：Stage 2 首次 smoke，梯度更新前  
性质：工程有效性硬约束修复，不使用 early、calibration、threshold、evaluation 或测试性能

## 触发证据

首次 smoke 在构造 DB Loss 时停止。原 Label-IF20 MILP 解包含 1,425 个 identity，FMB 正样本也为 1,425，因此 FMB 没有负样本。DB Loss 正确拒绝了这一输入，尚未进行任何模型梯度更新。

停止证据保留为：

- `outputs/route_a_frequency_reconstruction_v2/stage2_smoke/terminal_receipt_attempt1_aborted_manifest.json`
- `outputs/route_a_frequency_reconstruction_v2/stage2_smoke/seed_260821/LabelIF20/run_receipt_attempt1_aborted_manifest.json`
- `outputs/route_a_frequency_reconstruction_v2/stage1_data_audit/data_audit_attempt2_pretrain_invalid_if20.json`

## 最小修复

1. Label-IF20 的 seed、目标标签计数和 5% 标签误差容差保持不变。
2. 增加固定 `target_sample_count=1600`，使每个标签均同时存在正样本和负样本。
3. 修复后预期计数为 `[1425,75,824,475,258,137]`，实际 IF=19，仍在 IF20 的 5% 离散容差边界内；FMB 负样本数为 175。
4. Stage 1 新增统一硬门：FB、所有 Label-IF 和所有 Combination-IF manifest 的六个标签都必须同时具有正样本和负样本。
5. 不修改角色、数据 identity、功率视图、模型、损失权重、超参数、seed、评价角色或 frequency hard gate。

本修复只使用 optimization-role 标签的结构可训练性，不读取任何 evaluation prediction，更不访问 locked test、test-v2 或 test-v3。
