# Route A：VFT 严格匹配互补分数融合预注册方案（development-v3）

## 1. 单一可证伪假设

在严格双侧组合匹配协议下，冻结 F2 的 final-temporal 表征与固定 VFT-STFT 物理特征包含样本级互补证据。若该假设成立，不引入任何可学习融合参数的等权标准化分数融合，应同时优于两条单路分数以及保持边际分布但破坏样本对应关系的 STFT 打乱对照。

本轮只回答“互补性是否存在”，不训练统一分类器，不修改主干、损失、校准器或阈值，也不访问 test-v3。

## 2. 数据与证据边界

- Python：`D:\anaconda3\envs\PY10\python.exe`
- 数据：development-v3，SHA-256 必须为 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`
- 角色：沿用 C6 每个种子的 fit / validation / early，receiver identity 不跨角色
- 目标标签：仅 VFT（六标签索引 4）
- 样本：弱 VFT 正样本与 VFT 缺失负样本；两侧都只保留在本角色中同时出现的“其余五标签组合”
- 种子：2039、2040、2041、2042、2043
- 禁止数据：历史 locked test、test-v2、sealed test-v3

## 3. 冻结特征与分类器

### 3.1 Final-temporal 路

- 主干：每个种子已冻结的 `F2_DFMS_Full / C6_Joint`
- 输入：冻结主干 `forward_structure_maps` 的 final temporal map
- 汇聚：既有 `_temporal_summary`，维度 512
- 分类器：既有严格匹配实验的单层线性 probe，pos-weight BCE，checkpoint 完全冻结

### 3.2 VFT-STFT 路

- 输入：同一原始 I/Q 样本
- 固定变换：complex STFT，`n_fft=128`、`hop_length=32`、Hann window、双边频谱、频率 fftshift
- 特征：16×16 自适应时频网格及频率质心、扩展度、峰值轨迹统计，总维度 268
- 分类器：既有严格匹配实验的单层线性 probe，pos-weight BCE，checkpoint 完全冻结

两路均复用各自 checkpoint 内记录的特征均值和标准差，不重新训练或选择 checkpoint。

## 4. 无参数融合与打乱对照

对每路冻结 probe 的 logit，只用严格匹配 fit 样本计算均值与标准差：

`z = (logit - fit_mean) / max(fit_std, 1e-8)`

- 对齐融合：`score_aligned = 0.5 * z_final + 0.5 * z_stft`
- 打乱对照：`score_shuffled = 0.5 * z_final + 0.5 * permute(z_stft)`

打乱在每个角色内部独立进行，置换种子由实验种子和角色固定生成；保存置换数组 SHA-256。融合权重固定为 0.5/0.5，不搜索权重，不拟合门控器。

## 5. 指标与预注册晋级门槛

主要指标为 early 严格匹配 AUROC；同时报告 AUPR、validation 指标和 validation→early AUROC 差距。

五种子正式结果必须同时满足：

1. 对齐融合 early AUROC ≥ 0.620；
2. 对齐融合相对 final-temporal early AUROC 提升 ≥ 0.030；
3. 对齐融合相对 VFT-STFT early AUROC 提升 ≥ 0.010；
4. 对齐融合相对打乱对照 early AUROC 提升 ≥ 0.015；
5. 至少 4/5 种子同时实现“优于打乱对照”和“优于该种子的最佳单路”；
6. 对齐融合 validation→early AUROC 差距 < 0.150。

门槛在查看结果后不得修改。Smoke 只验证代码链路，不参与晋级判断。

## 6. 结果解释规则

- 全部门槛通过：只说明 VFT 两路冻结证据存在可复现的样本级互补性；下一轮才允许设计有界残差接回统一分类器，并配置同容量 MLP 与打乱分支对照。
- 未优于最佳单路：停止 VFT 融合，STFT 仅保留为独立诊断证据。
- 未优于打乱对照：判定提升不能归因于样本级 STFT 对齐信息，停止该机制。
- 仅绝对指标接近但门槛未全过：不得晋级，不得降低门槛。

## 7. 必须保存的产物

- 实验配置与门槛；
- development-v3、角色索引、C6 checkpoint、两路 probe checkpoint SHA-256；
- fit 分数标准化统计；
- 每角色严格匹配计数；
- 每角色样本 ID、标签、两路分数、对齐融合分数、打乱融合分数；
- 置换 SHA-256；
- 每种子 metrics/receipt 和五种子汇总；
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

