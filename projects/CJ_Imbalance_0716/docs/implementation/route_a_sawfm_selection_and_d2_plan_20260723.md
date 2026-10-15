# 路线 A：SawFM 检查点选择与 A6-D2 递进试验方案（2026-07-23）

## 1. 目标

上一轮 A6-PU 在 VFT、ISRJ 上稳定提升，但 SawFM 出现 macro-AUPR 提高、tail-AUPR 下降。下一轮先区分模型选择冲突与表征深度不足。

## 2. 阶段 S1：尾类检查点选择

保持 A6-PU 的训练过程不变：

- SawFM 尾类，seeds 2041、2042、2043；
- controlled Label-LT，N=2400，tail=400，IF=4；
- 从对应 Base-ASL 初始化；
- 只解冻最后时序块和分类头；
- 完整 DB；
- 分类头 lr=1e-3，最后块 lr=1e-4；
- 20 epochs；
- 每个 epoch 记录 SawFM AUPR；
- 使用 optimization-role SawFM AUPR 选择检查点。

S1 判据：相对 Base-ASL 的平均 SawFM AUPR 大于 0、至少 2/3 seeds 获胜，并且 macro-AUPR 平均不低于 -0.005。通过则说明主要问题是检查点选择，不执行 D2。

## 3. 阶段 S2：A6-D2

仅当 S1 失败时执行：

- 解冻 `backbone.temporal_backbone.3`、`backbone.temporal_backbone.5` 与分类头；
- 分类头 lr=1e-3；
- 最后时序块 lr=1e-4；
- 倒数第二时序块 lr=3e-5；
- 完整 DB；
- 不使用 L2-SP 和 TailMix；
- 使用 SawFM AUPR 选择检查点；
- 仍只运行 SawFM × 3 seeds。

S2 判据与 S1 相同。通过后才能回到 VFT、ISRJ 验证；失败则说明 SawFM 可能需要更早层或显式多尺度/类别条件化结构。

## 4. 治理

- Base-ASL、Base-DB、A6-PU-macro 均复用现有哈希检查点；
- 每个新运行必须共享对应 split SHA 与 manifest SHA；
- locked test 始终保持未使用；
- S1 与 S2 结果分别记录，不允许根据 S1 结果修改 S2 超参数。

