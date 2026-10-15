# Route A development-v3 RAL 损失基线预注册方案

状态：预注册；尚未运行 smoke 或正式五种子实验。

## 假设

ICCVW 2023 RAL 的多标签 `Ralloss` 是全局 logits/二值标签损失，可在不改变 F2、
RawSaw、安全路由或数据角色的前提下公平作为 Stage-E 长尾损失基线。该实验只检验
“已有的类别级鲁棒非对称损失能否解释弱分量收益”，不检验或选择 VTSE-VBF。

## 官方来源与等价移植

- 论文 DOI：`10.1109/iccvw60793.2023.00286`；作者仓库
  `WongiPark0628/RAL`，revision `0d7e7712c82ba9a36a08c888bb529885a871bc1f`。
- 固定公式为该 revision `models/get_optimizer.py::Ralloss`：
  `gamma_neg=4`、`gamma_pos=0`、`clip=0.05`、`eps=1e-8`、`lamb=1.5`、
  `epsilon_neg=0`、`epsilon_pos=1`、`epsilon_pos_pow=-2.5`。
- 原实现对一个 batch 的全部 label 取 `sum`。当前所有 development-v3 比较均以
  mean reduction 维持相同优化尺度，因此仅除以固定的 batch-label 元素数；这只
  是正的常数缩放，不改变同一 batch 中任意梯度方向或最优点。该归一化不改名为
  官方超参数，也不作搜索。

## 固定协议

| 项目 | 固定值 |
| --- | --- |
| 数据 | 已审计 development-v3，SHA 与现有 Focal-36 一致；fit/validation/calibration/threshold 角色不变 |
| 种子 | `2039, 2040, 2041, 2042, 2043`；smoke 仅 `2039` |
| 模型 | 相同 F2 初始化与部分解冻范围；RawSaw 仅覆盖 SawFM logit |
| 训练 | AdamW，36 epoch 上限，batch 64，LR `5e-4`，representation scale `0.1`，wd `1e-4`，clip 5 |
| 选择 | validation common macro-AUPR，其次 target weak AUROC；早停 patience 8 |
| 校准/阈值 | 与 Focal-36 完全相同的独立角色和每标签 FPR `<=0.20` 规则；不复用测试集 |
| 比较 | RAL−Focal-36 的五种子 paired bootstrap、方向统计、每标签弱/强/负指标 |

## 记录与停止规则

保存配置、源码 commit/blob hash、初始 F2/RawSaw/结果 checkpoint hash、history、
predictions、calibration、threshold、metrics 与 receipt；所有 receipt 都必须为
`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

若 smoke 的公式等价测试、角色/SHA/test-v3 seal 检查或训练链路失败，只修复工程
合同，不查看或比较任何性能。五种子结果无论优劣均仅作为主流损失基线：不改变
VTSE-VBF、融合、校准、阈值、test-v3 状态；不以结果反向搜索 RAL 参数。
