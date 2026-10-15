# Route A：ISRJ 周期专家组合适用域门控诊断预注册（development-v3）

## 研究问题

冻结周期专家接入 C4 后，严格匹配局部 AUROC 优势从约 +0.030 压缩到全分布弱 AUROC +0.006655。检验主要原因是否为：专家只在少数复合标签组合内有效，但固定不确定性残差被无条件施加到所有组合。

## 固定证据与边界

- 只使用 development-v3；
- 使用种子 2039--2043 已冻结的 C4 early predictions 和 ISRJ residual arrays；
- 弱分量定义沿用统一实验固定阈值：分量功率不高于 +10 dB；
- 不训练新参数，不修改主干、分类器、校准器、阈值或专家；
- oracle 使用真实其他标签，仅作为不可部署上界；
- test-v3 保持 `access_authorized=false, analysis_allowed=false, used=false`。

## 组合域定义

对每个种子，仅从 fit role 的 `ISRJ=1 AND weak=1` 样本读取其余五个标签，取唯一二值组合集合。禁止从 early role 选择组合。

## 比较方案

- D0：C4 原始 ISRJ logit；
- D1：无组合门控的对齐周期残差；
- D2：真实其他标签恰好属于 fit 组合集合的 oracle gate；
- D3：根据冻结 C4 其余五标签校准概率计算组合集合的总似然，作为 soft gate；
- D4：根据冻结 C4 其余五标签最终预测计算 exact-match hard gate；
- D5：soft gate 下的 block-shuffle 周期残差。

所有比较只计算 early role 的排序指标；不在 early 上重新选择阈值。

## 指标

- ISRJ 弱正样本对全部负样本 AUROC（主指标）；
- ISRJ 全正样本 AUPR；
- oracle/soft/hard gate 在弱正、域内负样本、域外负样本上的平均覆盖；
- 五种子逐种子差值和 bootstrap 95% CI。

## 判定逻辑

1. 若 D2 相对 D1 的弱 AUROC均值不足 +0.010，否定“组合域稀释是主要原因”，停止该门控方向；
2. 若 D2 达标但 D3 相对 D1 不足 +0.005，或未在 4/5 种子为正，说明 oracle 存在但 C4 上下文预测不足，不训练融合器；
3. 只有 D3 回收至少 50% 的 oracle 增益、相对 D5 弱 AUROC至少 +0.005、且 4/5 种子为正，才允许预注册低容量可部署上下文融合器；
4. D4 仅用于判断硬组合错误是否比软门控严重，不参与晋级。

## 产物

- 每种子组合集合、覆盖率、逐方案排序指标、差值；
- 汇总 JSON、配置、哈希和正式结果报告；
- 明确 `locked_test_used=false, test_v3_used=false`。

实现前 smoke 审计曾发现脚本草稿把弱阈值误设为 -5 dB；该 smoke 作废。正式运行前已修正为预注册和统一实验一致的 +10 dB，未据作废结果调整任何模型参数或判定门槛。
