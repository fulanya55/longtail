# 路线 A：PAM-IQ V2 双路径标签查询网络与分阶段试验方案（2026-07-24）

## 1. 目的与冻结边界

本阶段修复 PAM-IQ 对显著局部结构建模较强、但弱分量证据保留不足的问题。整改不再增加更多卷积尺度，而是围绕四个可归因机制展开：

1. 增加不经过注意力的全局保真支路；
2. 将样本级单一时间注意力改为六标签独立查询；
3. 将乘法注意力改成带恒等旁路的残差注意力；
4. 将标签频次目标与弱可见性目标分阶段解耦。

本轮只使用 development roles，`locked_test_used=false`。第一阶段只改变网络结构，保持 DB Loss、训练数据和主要选点规则不变；结构通过后才允许加入弱正排序损失和功率配对。

## 2. 输入与共享编码器

输入保持为 RMS 归一化复数 I/Q：

```text
[B, 2, 3000]
```

共享编码器沿用 PAM-IQ：

| 模块 | 输出 | 固定参数 |
|---|---|---|
| Complex residual block 1 | `[B,16,3000]` real/imag | kernel=7 |
| Complex average pool | `[B,16,1500]` real/imag | kernel=2, stride=2 |
| Complex residual block 2 | `[B,32,1500]` real/imag | kernel=5 |
| Phase interaction fusion | `[B,64,1500]` | real/imag/magnitude/cosΔφ/sinΔφ，1×1 projection |
| Temporal stage 1 | `[B,64,750]` | stride=2；depthwise `(k,d)=(3,1),(7,2),(15,4)` |
| Temporal stage 2 | `[B,96,375]` | stride=2；同上 |
| Temporal stage 3 | `[B,128,375]` | stride=1；同上 |

最终时序 token 记为：

```text
H ∈ R[B,128,375]
```

## 3. 注意力整改

### 3.1 残差注意力

V2 中 ECA、PCAF 和 GlobalContext 不再直接替换输入，而使用：

```text
y = x + gamma * (Attention(x) - x)
```

`gamma` 为可学习标量，初始化为 `0.10`。位置保持为：

- stage 1：ECA(k=5)；
- stage 2：PCAF(k=5, dilation=1/3/5, reduction=8)；
- stage 3：GlobalContext(reduction=8)。

该形式保证初始化时至少保留 90% 恒等特征，避免连续乘法门控过早删除低响应证据。

### 3.2 双路径池化候选 D1

全局保真路径：

```text
H -> unweighted mean + unweighted std -> [B,256]
  -> Linear(256,128) -> LayerNorm -> GELU -> Dropout
  -> g ∈ R[B,128]
```

局部注意力路径继续使用 attentive mean/std：

```text
H -> attentive mean + attentive std -> [B,256]
  -> Linear(256,128) -> LayerNorm -> GELU -> Dropout
  -> a ∈ R[B,128]
```

融合：

```text
f = g + tanh(gamma_pool) * a
```

`gamma_pool` 初始化为 `0.10`，最终 `feature_dim=128`。

## 4. 双路径标签查询候选 Q1

Q1 共享 D1 的无注意力全局保真路径，但局部路径改成六标签查询。

### 4.1 标签查询池化

固定参数：

| 参数 | 数值 |
|---|---:|
| 标签数 | 6 |
| query dimension | 32 |
| key projection | Conv1d(128,32,1) |
| value projection | Conv1d(128,32,1) |
| query embeddings | `[6,32]`，Normal(0,0.02) |
| attention floor | 0.20 |
| query normalization | LayerNorm(32) |

每标签注意力为：

```text
a[c,t] = softmax(q[c] · key[t] / sqrt(32))
a_floor[c,t] = 0.20 / T + 0.80 * a[c,t]
z[c] = sum_t a_floor[c,t] * value[t]
```

得到：

```text
Z ∈ R[B,6,32]
```

公开 backbone 特征为：

```text
concat(g, flatten(Z)) -> [B, 128 + 6×32] = [B,320]
```

### 4.2 分类器参数

分类器由全局主头和标签查询残差头组成：

```text
global_logits = Linear(128,6)(g)
query_logit[c] = dot(z[c], w_query[c]) + b_query[c]
logit[c] = global_logits[c] + gamma_query[c] * query_logit[c]
```

固定参数：

| 参数 | 数值 |
|---|---:|
| global classifier | Linear(128,6) |
| query classifier weights | `[6,32]` |
| query classifier bias | `[6]` |
| query residual scale | `[6]`，初始化 0.10 |
| classifier dropout | 第一阶段 0.20；第二阶段 0.30 |

查询头保持低秩共享结构，不建立六套大 MLP。全局主头提供安全边界，查询头只学习标签特异残差。

## 5. 损失函数

### 5.1 第一阶段结构筛选

所有候选统一使用现有完整 DB Loss：

```text
L_stage1 = L_DB
```

不得在结构筛选中混入 ASL、弱正权重、功率配对或 prototype，以避免归因混杂。

### 5.2 第二阶段弱可见性目标（仅在 Q1 通过后）

弱正位置由训练集参数计算：标签存在，且相对同样本最强正分量不高于 `-10 dB`。

主目标保持 DB，新增弱正—负样本排序：

```text
L = L_DB + lambda_weak * L_weak_rank

L_weak_rank = softplus(margin - logit_weak_positive + logit_sampled_negative)
```

预注册参数：

- `margin=0.20`；
- 每个弱正位置采样最多 4 个同标签负样本；
- `lambda_weak ∈ {0.10, 0.25, 0.50}`；
- 前 20% update 只训练 DB，之后在 20%～40% update 线性升高弱正损失；
- 第一轮只让 `L_weak_rank` 更新 query residual head；通过后才允许回传最后一个 temporal stage。

不使用“强正高于弱正”的排序，避免通过压低弱正 logit 满足约束。

## 6. 阈值与校准

第一阶段所有模型同时报告：

1. fixed 0.5；
2. global threshold；
3. per-label F1；
4. per-label F2；
5. shrinkage per-label F1；
6. per-label Youden。

结构晋级的主要排序指标仍为 macro-AUPR；主要决策指标暂保持 shrinkage-F1，保证与 B1/P0 历史结果可比。per-label F2 作为弱召回诊断，不用于第一阶段自动晋级。

第二阶段阈值冻结规则：

```text
在 threshold-selection role 上，寻找 weak recall 最高的逐标签阈值；
约束：Hamming <= B1 + 0.005，macro-F1 >= B1 - 0.010。
```

若没有候选满足约束，则回退到 per-label F1。温度校准第一阶段继续使用共享标量温度；逐标签正则化温度只作为结构通过后的独立消融，禁止与 Q1 首轮同时加入。

## 7. 优化超参数

### 7.1 第一阶段公平筛选

为与 B1/P0 配对，第一阶段所有模型使用相同协议：

| 参数 | 数值 |
|---|---:|
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| batch size | 64 |
| epochs | 20 |
| patience | 20 |
| warmup | 10% |
| gradient clip | 5.0 |
| AMP | 开启 |
| selection | macro-AUPR |
| train samples | 2400 |
| eval role max samples | 300 |
| dropout | 0.20（所有模型一致） |

### 7.2 Q1 通过后的优化筛选

仅对通过结构门槛的 Q1 测试：

| 参数 | 候选 |
|---|---|
| backbone lr | `1e-4`, `2e-4` |
| query/global head lr | `3e-4` |
| weight decay | `5e-4`, `1e-3` |
| dropout | `0.30` |
| warmup | 20% |
| epochs | 30 |

第二阶段采用词典序 checkpoint：先要求 macro-AUPR 距离该 run 最优值不超过 0.005，再在候选 epoch 中最大化 weak-component recall，最后最小化 Hamming loss。

## 8. 第一阶段正式矩阵

| 编号 | backbone | 目的 |
|---|---|---|
| B1 | `complex_1d_true`, width=56 | 容量匹配强对照 |
| P0 | `phase_multiscale`, width=16 | 当前 PAM-IQ 锚点 |
| D1 | `pam_iq_dual_pool`, width=16 | 验证无注意力全局旁路 |
| Q1 | `pam_iq_dual_query`, width=16 | 验证标签查询与残差分类头 |

实际部署参数量：B1=260,630，P0=253,325，D1=286,481，Q1=257,819。Q1 与 B1 相差约 1.1%，属于主要容量匹配候选；D1 参数量高约 9.9%，只作为机制诊断，不承担最终晋级。

固定 seeds：`2041, 2042, 2043`。同 seed 四个模型必须共享 split SHA256 和 manifest SHA256。

## 9. Q1 晋级门槛

Q1 相对 B1 必须同时满足：

1. 平均 delta macro-AUPR >= -0.005；
2. 平均 delta SawFM-AUPR >= +0.010；
3. 平均 delta weak-component recall >= +0.030；
4. 平均 delta Hamming loss <= +0.005；
5. FMB、SinFM、MFT、VFT 任一标签平均 AUPR 下降不超过 0.030；
6. 至少 2/3 seeds 的 weak-component recall 为正增益。

D1 不承担最终晋级，只用于判断全局旁路是否有效。若 D1 改善而 Q1 退化，说明标签查询分类头仍需整改；若 D1/Q1 均无改善，则返回更早的特征门控和弱正监督方向。

## 10. 执行顺序与产物

1. 完成代码、shape/gradient/freeze 测试；
2. 单 seed、1 epoch、每角色最多 64 样本 GPU smoke；
3. 正式执行 4 models × 3 seeds；
4. 保存每轮 history、最优模型、温度、所有阈值模式、预测、manifest、split hash 和 receipt；
5. locked test 继续冻结；
6. 第一阶段通过后另建第二阶段输出目录，不允许覆盖第一阶段结果。

计划输出：

```text
outputs/tapc/route_a_pam_iq_v2_structure_screen/
docs/results/route_a_pam_iq_v2_structure_screen_results_20260724.md
```

## 11. 执行状态（2026-07-24）

- 结构、分类器、F2 阈值支持和正式矩阵已实现。
- 定向测试 `21 passed`；全量测试 `143 passed, 36 warnings`。
- GPU smoke 4/4 完成；正式试验 12/12 完成。
- locked test 未使用。
- Q1 的六项自动晋级检查全部未通过。
- 按本文件预注册规则，第二阶段弱正排序损失没有执行。

完整结果与失败归因见：

`docs/results/route_a_pam_iq_v2_structure_screen_results_20260724.md`
