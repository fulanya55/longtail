# Route A Frequency Reconstruction V2 种子语义执行补充

日期：2026-08-08  
适用阶段：Stage 1 数据审计  
性质：字段语义澄清，不改变预注册样本、角色、manifest、模型或门槛

## 发现

预注册文件把 `3930000001..3930025200` 简写为 receiver identity range。生成器实际将这一区间保存为 manifest 的 `sample_seed`；MAT 文件顶层保存的是由它确定性派生的 `receiver_noise_seed`：

`receiver_noise_seed = (sample_seed + 2 * 2246822519) mod 2^32`

该映射在 uint32 域上一一对应。因此本轮 MAT 文件中的 `receiver_noise_seed` 范围为 `4128677743..4128702942`，逆映射恢复出的 `sample_seed` 范围仍精确为 `3930000001..3930025200`。

## 审计规则

1. 以 formal generation receipt 中的完整 sample-seed signature 和从 MAT `receiver_noise_seed` 逆映射得到的 sample seed 双重核验预注册范围。
2. 历史碰撞仍按 Stage 0 的 sample-seed 范围审计；由于映射为双射，sample-seed 无碰撞等价于 receiver-noise-seed 无碰撞。
3. VH/VL 仍必须逐 identity 具有相同的 labels、receiver noise seed、component seeds、目标字段和顺序。
4. 本补充不授权修改 IF、角色、损失、模型、训练预算、评价角色或 hard gate。

## 首次审计保留

首次 Stage 1 审计按错误的字段直读假设停止，收据和审计分别保留为：

- `outputs/route_a_frequency_reconstruction_v2/stage1_data_audit/run_receipt_attempt1_stopped.json`
- `outputs/route_a_frequency_reconstruction_v2/stage1_data_audit/data_audit_attempt1_stopped.json`

重新审计只修正字段语义和 5% 边界的浮点比较容差，不改变任何数据选择结果。
