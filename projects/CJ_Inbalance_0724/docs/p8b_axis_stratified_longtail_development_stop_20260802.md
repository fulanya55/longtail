# P8b：轴分层批采样的三种子开发止损记录

## 结论

P8b 未通过开发门，永久停止为“在该冻结 EfficientNet-B1/BCE 训练契约中，固定轴分层批成员构成可改善多轴长尾 VFT 识别”的候选机制。结论是 `stop_no_confirmation_data`：不构造两个额外模型种子、不生成新身份确认集，也不申请或读取任何测试集。

这不是只由阈值造成的表面差异。三个种子的 10 dB VFT AUPR 都低于随机批 C0，均值差为 -0.025064；同时最坏功率下 VFT FPR 均明显超出 +0.02 风险上限。因而它既没有带来排序收益，也出现了不安全的响应/误报放大。

## 冻结的单变量问题

- **C0**：随机 replacement batch，BCE。
- **C1**：唯一改变为确定性 batch 成员配额：弱 VFT 16、其他 VFT 4、VFT 困难负样本 16、其余 28；BCE 与 C0 相同。
- 两臂均为 `efficientnet_b1_1d`（dropout 0.2，head 256），AdamW（lr `3e-4`、weight decay `1e-4`），batch 64，30 epoch × 38 updates = 1,140 optimizer updates。
- 早停只按 early-stop 角色的 macro-AUPR、VFT AUPR、最早 epoch；F1 阈值只在 calibration 角色确定；开发报告使用 held development 角色的九功率预测。

## 数据与证据边界

新的训练根为 `generated_data/p8b_axis_stratified_longtail_seed1310000000`：63 个组合各 400 个新身份、共 25,200 个身份，九功率 10--50 dB 对齐。独立审计通过：九功率身份/标签对齐、仅训练路径、与 P8a 零身份交集、精确身份数及每组合 400 条。

开发模型种子固定为 18101、18102、18103，每个种子均完成 C0/C1 一对。`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 预注册门与实测差值（C1 - C0）

| 指标 | seed 18101 | seed 18102 | seed 18103 | 均值 / 门 | 判定 |
|---|---:|---:|---:|---:|---|
| 10 dB VFT AUPR | -0.017972 | -0.012536 | -0.044682 | -0.025064；每种子非负且均值 >= +0.02 | 失败 |
| 10 dB macro-AUPR | -0.009814 | -0.001599 | -0.011345 | -0.007586；每种子 >= -0.005 | 失败 |
| 10 dB Hamming loss | +0.015873 | -0.003439 | +0.006151 | +0.006195；每种子 <= +0.005 | 失败 |
| 10 dB VFT 困难负 FPR | -0.012500 | +0.106250 | +0.150000 | +0.081250；每种子 <= +0.02 | 失败 |
| 九功率最坏 VFT FPR | +0.280645 | +0.299194 | +0.083871 | +0.221237；每种子 <= +0.02 | 失败 |
| 10 dB VFT 正/困难负分离 | +0.076013 | +0.034405 | -0.121262 | -0.003615；每种子 >= 0 | 失败 |

非 VFT 标签的系统性 FPR 上升门也失败：seed 18101 的两个非 VFT 标签分别上升 +0.071774 与 +0.220968。三个种子的 VFT AUPR bootstrap 95% CI 分别为 [-0.038543, +0.002820]、[-0.037202, +0.013445]、[-0.066510, -0.024251]；它们不支持正的可重复排序增益。

## 失败归因

1. 固定配额并未改善 VFT 的排序：三种子 AUPR 均下降，故不能解释为“召回换精度”的可接受权衡。
2. 配额在个别功率/种子上显著扩大了 VFT 的负类响应，最坏功率 FPR 上升 +0.084 至 +0.299，超过预注册上限 4--15 倍。这满足“FPR 恶化即响应放大失败”的判定规则。
3. 正/困难负分离没有稳定性：虽然两个种子表面上升，第三个种子下降 -0.121262，且均值为负；因此没有机制性证据表明模型学会了组合关系。

## 机械性回执修复

六个训练分支完成后，原 runner 仅在最终 `json.dumps` 时因 manifest 内的 `numpy.ndarray` 停止；训练、checkpoint 和每功率开发预测均已保存。新增 JSON 标准类型转换单元测试（4 passed），并使用 `cj0724/finalize_p8b_axis_stratified_longtail.py` 从这六个不可变 checkpoint 及其保存的开发预测恢复指标和回执；未重新训练、未改变任何网络、数据、阈值或选择规则。`result.json` 同时记录原冻结训练 runner hash 与 finalizer hash，以避免把该输出层修复误记为模型改动。

## 不允许的后续动作

- 不调整 P8b 的配额、采样步数、损失、骨干、尾部定义或阈值以追求通过；
- 不扩展至五个模型种子或新身份确认；
- 不生成、读取或复用开发外测试与 locked test。

任何后续多轴长尾方法必须以新的预注册、独立训练身份及独立的等预算强基线开始，且先用排序、困难负 FPR 和跨功率最坏风险证明其不是响应放大。

## 可复核产物

- 预注册：`configs/p8b_axis_stratified_longtail_preregister_20260802.json`
- 冻结：`configs/p8b_axis_stratified_longtail_method_freeze_20260802.json`
- 数据审计：`outputs/p8b_axis_stratified_longtail_preflight_20260802/data_audit.json`
- 三种子开发回执：`outputs/p8b_axis_stratified_longtail_development_20260802/result.json`
- C0/C1 checkpoints 与九功率开发预测：同一 development 输出目录
