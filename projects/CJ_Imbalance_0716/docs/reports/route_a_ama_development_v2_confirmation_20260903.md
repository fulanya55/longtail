# AMA 六类 IF3 / 真 Anchor development-v2 五种子确认

## 数据角色

- IF3 fit：来自同协议九类 `closedset_9class_test.mat` 的纯 AMA 六类样本，N=8000，IF=3.0，CF=7.8537，63 组合。
- Anchor fit：来自同协议九类 `closedset_9class_train.mat`，排除上一轮真实 Anchor 已用身份后无放回抽样，N=8000，IF=1.0，CF=1.0079，63 组合。
- 公共 validation：N=2016，63 组合，每组合 32 个，IF=CF=1。
- 公共 development-test：N=4032，63 组合，每组合 64 个，IF=CF=1。
- 所有 receiver identity 交集为 0；没有 locked test；E5 未修改。

## 固定训练协议

`F2_DFMS_Full`，真实 I/Q `[B,2,3000]`，BCEWithLogitsLoss，AdamW，lr=`1e-4`，weight decay=`1e-4`，batch=32，8 epochs，梯度裁剪 5，最佳 epoch 按 validation Macro-AUPR。

## 五种子结果

| 条件 | Val Macro-AUPR | Val Macro-F1 | Val 组合准确率 | Val Hamming | Dev-test Macro-AUPR | Dev-test Macro-F1 | Dev-test 组合准确率 | Dev-test Hamming |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF3 | 0.979576 ± 0.002272 | 0.917139 ± 0.002147 | 0.609226 ± 0.012288 | 0.081481 ± 0.002592 | 0.977965 ± 0.002182 | 0.913943 ± 0.003299 | 0.605060 ± 0.015907 | 0.084185 ± 0.003521 |
| True Anchor | 0.984417 ± 0.000452 | 0.935541 ± 0.001580 | 0.693056 ± 0.008680 | 0.062252 ± 0.001135 | 0.982292 ± 0.000726 | 0.931727 ± 0.002550 | 0.679663 ± 0.007540 | 0.065592 ± 0.001978 |

配对差异（IF3−Anchor，5 个相同种子）：

- Validation Macro-AUPR：`−0.004841`
- Validation Macro-F1：`−0.018402`
- Validation 组合准确率：`−0.083829`
- Validation Hamming：`+0.019229`
- Dev-test Macro-AUPR：`−0.004327`
- Dev-test Macro-F1：`−0.017784`
- Dev-test 组合准确率：`−0.074603`
- Dev-test Hamming：`+0.018593`

五个种子的所有主要方向一致，说明在新的身份集合和公共均衡评估集上，IF3 的标签长尾确实造成稳定退化，尤其集中在组合一致性和 Hamming 错误，而不仅仅是概率排序指标。

## 解释边界

这是 development-v2 确认性证据，不是 locked-test 结论。IF3 fit 使用 source-B 文件，Anchor fit 使用 source-A 文件；两者生成协议和科学语义版本一致，但仍存在文件/身份域差异，因此该结果不能单独视为纯标签 IF 的最终因果估计。下一阶段长尾损失比较必须固定本 manifest、公共 validation/test 和全部训练预算。

## Receipts

- Manifest: `outputs/route_a_ama_development_v2_20260903/manifest.json`
- Summary: `outputs/route_a_ama_development_v2_20260903/summary.json`
- Terminal receipt: `outputs/route_a_ama_development_v2_20260903/terminal_receipt.json`
