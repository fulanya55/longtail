# 路线 A：IF=4 ASL–DB 混合损失技术与实验方案（2026-07-23）

## 1. 问题定位

受控 Label-LT 试验表明，DB 在 IF=2/3 下稳定优于 ASL，但在 IF=4 下出现以下分裂：

- macro-AUPR：+0.01434；
- 指定尾类 AUPR：-0.01633；
- macro-F1：-0.01161；
- Hamming loss：+0.00667。

非尾类平均 AUPR 提升约 +0.02456，说明 DB 仍包含有效排序信号；直接放弃 DB 会丢失该部分收益。下一步采用 ASL–DB 混合损失，目标是降低 DB 在极端共现长尾下的梯度强度，同时保留其非尾类排序收益。

## 2. 改进方法

定义

\[
L_{mix}(\lambda)=(1-\lambda)L_{ASL}+\lambda L_{DB}.
\]

试验两个预注册权重：

- Mix-0.25：`λ=0.25`，偏向尾类保护；
- Mix-0.50：`λ=0.50`，平衡 ASL 与 DB。

不在看到结果后搜索连续权重，避免在 3 seeds 上过拟合。

## 3. 数据与训练协议

- 数据：0716 四类 optimization role；
- 协议：受控 Label-LT，名义 IF=4，实际 IF=3.8；
- 总训练样本：1600；
- 尾类目标正例：400；
- 标签容差：5%；
- 尾类：FMB、MFT、ISRJ；
- seeds：2036、2037、2038；
- 网络：T0；
- epochs/patience：20/20；
- 模型选择：macro-AUPR；
- 主阈值模式：per-label F1；
- 锁定测试：禁止使用。

FMB 是上一轮的正向 sanity check；MFT、ISRJ 是主要失败标签。ASL、DB 的 18 次既有结果直接复用，新训练为 `2 mix × 3 tails × 3 seeds = 18` 次。

## 4. 判定门槛

候选混合损失必须同时满足：

1. 对 MFT 和 ISRJ，tail-AUPR 相对 DB 均提高；
2. 对 MFT 和 ISRJ，tail-AUPR 相对 ASL 的平均差不低于 -0.005；
3. 对 MFT 和 ISRJ，macro-AUPR 相对 ASL 为正；
4. per-label F1 模式下 macro-F1 相对 ASL 不低于 -0.005；
5. Hamming loss 相对 ASL 的恶化不超过 +0.005；
6. FMB sanity check 的 macro-AUPR 相对 ASL为正，tail-AUPR 不低于 -0.005。

若两个候选均通过，以 MFT/ISRJ 中较差的 tail-AUPR 差值优先，再比较平均 macro-AUPR。若均不通过，则停止损失插值路线，进入表征与分类器解耦训练。

## 5. 实现与输出

- `cjlt/losses/classification.py`：实现 `ASLDBMixedLoss`；
- `experiments/run_tapc.py`：增加 `asl_db_mix` 和 `--db-mix-weight`；
- `experiments/run_route_a_if4_mix.py`：训练、复用基线、配对分析和门槛判定；
- 输出：`outputs/tapc/route_a_if4_mix/`。

所有新运行必须与对应 ASL/DB 基线共享相同的 split 和 manifest 哈希。
