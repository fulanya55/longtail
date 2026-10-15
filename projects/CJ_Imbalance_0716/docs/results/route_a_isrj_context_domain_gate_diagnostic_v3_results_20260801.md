# Route A：ISRJ 周期专家组合适用域门控诊断结果（development-v3）

## 1. 判定

组合适用域门控假设被否定，`deployable_fusion_permission_gate.passed=false`。不应训练或接入基于 fit exact-composition 的门控器。

五个随机种子中，真实标签 oracle、C4 软预测门控和 C4 硬预测门控相对无门控周期残差均为负。即便知道真实其他五标签，按 fit 中出现的精确组合限制周期专家，也会删除大量有效弱 ISRJ 证据。

## 2. 核心结果

| 方案 | ISRJ AUPR | 弱正 vs 负样本 AUROC |
|---|---:|---:|
| D0 C4 | 0.842995 | 0.566456 |
| D1 无组合门控对齐残差 | **0.850065** | **0.573111** |
| D2 真实组合 oracle | 0.843566 | 0.566941 |
| D3 C4 软组合门控 | 0.843437 | 0.566907 |
| D4 C4 硬组合门控 | 0.843479 | 0.567308 |
| D5 软门控 block-shuffle | 0.843059 | 0.566627 |

相对 D1：

- oracle：弱 AUROC −0.006169，5/5 种子为负；
- soft：弱 AUROC −0.006203，5/5 种子为负；
- hard：弱 AUROC −0.005803，5/5 种子为负。

D3 相对 D5 只有 +0.000280，虽然 5/5 为正，但远低于预注册 +0.005 门槛。

## 3. 原因

fit role 的弱 ISRJ 正样本只有两种精确其他标签组合：

- `[FMB=1, MFT=0, VFT=0, SawFM=0, SinFM=0]`；
- `[FMB=1, MFT=0, VFT=0, SawFM=1, SinFM=0]`。

这两个 fit 组合在 early 弱 ISRJ 正样本中的 oracle 覆盖率仅为 3.37%--6.69%。严格匹配局部实验在 validation/early role 内分别建立当期可比较组合，因此它验证的是周期特征跨复合上下文的可迁移排序能力，而不是“只对 fit 两种组合有效”。

无条件周期残差优于所有 exact-context gate，说明周期性属于相对标签不变的物理证据。用组合身份限制它，反而把可泛化的专家错误地变成窄域专家。

## 4. 对后续设计的约束

停止：

- fit exact-composition oracle/soft/hard gate；
- 基于其他五标签组合的专家路由；
- 继续扫描组合门控阈值或概率温度。

保留：

- 冻结 310 维周期 HGB；
- block-shuffle 因果控制；
- 无条件接入的 5/5 正向证据。

下一步应解决“固定残差不能充分利用专家分数”而非“专家适用域”。风险最低的验证是冻结 C4 与周期专家，只训练一个极低容量 ISRJ 分数融合头，在全 fit 分布上直接学习 C4 logit、周期 score 和二者交互；同时用相同结构的 block-shuffle score 作为容量控制。只有该头在五种子统一指标上超过固定残差并通过控制，才允许并入联合候选。

## 5. 证据边界与产物

- 证据级别：五种子 development-v3 冻结诊断；
- 学习参数：0；
- test-v3：未访问、未分析、未使用；
- 预注册：`docs/plans/route_a_isrj_context_domain_gate_diagnostic_v3_plan_20260801.md`；
- 实现：`experiments/run_route_a_isrj_context_domain_gate_diagnostic_v3.py`；
- 测试：`tests/test_route_a_isrj_context_domain_gate_diagnostic_v3.py`（3 passed）；
- 输出：`outputs/tapc/route_a_isrj_context_domain_gate_diagnostic_v3`；
- 汇总 SHA256：`77E3A9C0E685F56E8FFAABB344F3BCDB851C8AD366F8968173A28F4E0A3B626C`。

实现审计记录：首次 smoke 前发现 Windows 下 sklearn/torch DLL 加载顺序问题并修复；首次 smoke 又发现弱阈值草稿误设为 −5 dB，该 smoke 作废并在正式运行前改回统一协议的 +10 dB。正式五种子结果未用于调整模型或门槛。

