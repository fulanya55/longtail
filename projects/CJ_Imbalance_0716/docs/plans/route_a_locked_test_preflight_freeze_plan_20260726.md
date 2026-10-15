# 路线 A：locked-test 前冻结审计计划（2026-07-26）

## 目标

在不读取 locked test 的前提下，把 `RawSaw+H2Others` 的网络、checkpoint、校准参数、阈值、标签顺序、预处理和代码版本固化为可唯一重放的推理包，并验证所有 development 证据和身份审计未漂移。

## 当前缺口

五种子正式实验已经保存Raw统一阈值、预测和checkpoint哈希，但Raw Platt的系数与截距没有作为独立冻结产物持久化。若最终测试时重新拟合Platt，会使“校准已冻结”无法审计。因此必须先在原ordinary calibration角色上确定性重放，并保存每seed的Platt参数。

## 执行步骤

1. 修改五种子development脚本，只增加`calibration.json`输出，不改变网络、数据角色、训练、校准、阈值或指标；
2. 使用PY10重放五种子development，确认正式summary SHA256保持不变；
3. 实现preflight工具，逐seed验证：
   - F2/H2/Raw checkpoint存在且SHA256一致；
   - split、calibration selection和development selection哈希存在；
   - identity overlap为0；
   - H2 temperature、H2逐标签threshold、Raw Platt和Raw threshold均已持久化；
   - 五种子确认门和标签路由门仍通过；
   - 所有receipt均为`locked_test_used=false`；
4. 生成`frozen_inference_bundle.json`、`preflight_audit.json`和`run_receipt.json`；
5. 运行单元测试和全量回归测试。

## 冻结内容

- 类别顺序：`[FMB, SawFM, SinFM, MFT, VFT, ISRJ]`；
- 输入：`[B,2,3000]`，complex RMS normalization；
- F2、H2、Raw网络结构参数；
- 五个seed的checkpoint绝对路径和SHA256；
- H2 temperature和六标签threshold；
- Raw Platt coefficient/intercept和SawFM threshold；
- 代码、配置、正式summary的SHA256；
- `SawFM=Raw+Platt`、其他五类`=H2`的路由规则；
- 明确`ensemble=false`和`locked_test_authorized=false`。

## 通过门槛

1. 正式summary SHA256仍为`1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC`；
2. 5/5 checkpoint哈希一致；
3. 5/5校准与阈值产物完整；
4. identity overlap全部为0；
5. confirmation满足4/5，label routing满足5/5；
6. locked test未访问；
7. 全量测试通过。

## 停止规则

- 任一checkpoint、split、阈值或summary哈希漂移：停止并不得申请locked test；
- Platt重放导致正式指标或summary变化：停止并调查非确定性；
- preflight通过后仍只标记`ready_for_authorization`，不能自动执行locked test。

