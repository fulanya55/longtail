# Route A development-v3 主流基线证据审计

## 1. 审计结论

阶段 E 尚未完成。项目内存在大量 ASL、DB、ASL-DB、解耦头和多任务优化实验，但大部分使用旧 label-LT、development-v2 或新的 development-v6 科学问题，不能直接作为当前 development-v3 统一 VTSE-VBF/F3 的公平论文基线。

当前 development-v3 可计入的公平对照只有：

1. `C6_Joint`：冻结 F2 初始化后，部分解冻 shared stage 与 774 参数六标签分类器，pos-weight BCE，12 epochs；
2. `C4_SameCapacity_NoVisibility`：与统一 VTSE 候选相同新增参数量，但使用 all-positive 普通监督；
3. `ThreeStratumBalancedHead`：同容量三层等权头，已失败；
4. `F2_C4OnlyWeakWeightedHead`：冻结标量 C4 score 的弱正加权重标定控制，排序不变；
5. `F4_BlockShuffledFusionHead`：与 F3 同参数量、同训练方式但破坏长周期顺序的物理容量控制。

这些对照足以支持部分机制归因，但不足以满足 Goal 规定的主流方法比较。

## 2. 当前 development-v3 基线状态

| 方法 | 当前 v3 公平证据 | 能回答的问题 | 不能替代的证据 |
|---|---|---|---|
| pos-weight BCE / C6 | 已完成，五种子 | 标签正负不均衡的标准分类边界 | 普通 BCE、Focal、ASL、DB |
| C4 all-positive capacity | 已完成，五种子 | 参数量与强正锚点贡献 | 主流损失比较 |
| Three-stratum equal weight | 已完成，五种子，失败 | 仅靠弱/强/负重加权是否充分 | 完整 visibility expert |
| C4-only weak-weight scalar head | 已完成，五种子 | F3 是否只是重标定/弱正加权 | 表征级损失基线 |
| block-shuffle F4 | 已完成，五种子 | 周期顺序相对局部统计的贡献 | 同容量普通时序 MLP |

## 3. 不能迁移为当前证据的历史实验

### 3.1 旧 ASL/DB/CB-ASL 路线

旧 `route_a_label_lt_losses`、六类 Base-DB、ASL-DB mix 和反向解耦实验使用不同数据、manifest、训练规模、角色或阈值协议。它们证明 DB 在标签频次长尾上的历史价值，但不能证明 DB 在 development-v3 标签内可见性问题上弱于 F3。

### 3.2 development-v6 多任务优化

`route_a_development_v6_multitask_baselines` 比较 PCGrad、MGDA、CAGrad、GradNorm 等两任务优化方法，使用新的 25,200 身份和 task-vector 科学问题。它是独立研究线，不能替代 BCE/Focal/ASL/DB/CB/LA 的 v3 公平比较，也不能与 F3 直接合并统计。

### 3.3 历史两阶段与冻结头

旧 DB representation + ASL head、H2 PosBCE 和 IF4 decoupled head 使用旧模型与 split。可用于解释“表示与分类边界协同”，但不能作为当前 F3 的最终数值基线。

## 4. 必须新增的公平实验

下一实验固定使用：

- development-v3 SHA256 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 五种子 2039--2043 的完全相同 role_indices；
- 同一 F2 checkpoint 初始化；
- 与 C6 相同 trainable shared scope、分类器、batch 顺序、AdamW、学习率、12 epochs、模型选择规则；
- 相同 calibration role、FPR<=0.20 threshold role 和 early evaluation；
- RawSaw SawFM 路由作为所有候选共同固定模块，避免 F3 因独享 RawSaw 获得宏观优势；
- test-v3 禁止访问。

首批必须比较：

1. BCE；
2. pos-weight BCE；
3. Focal；
4. ASL；
5. DB；
6. Class-Balanced BCE；
7. Logit Adjustment BCE；
8. 两阶段解耦分类器；
9. 与 F3 新增参数量匹配的普通时序 MLP。

每个方法必须保存 checkpoint、history、校准、阈值、predictions、metrics、参数量、训练时间和哈希。不得使用各方法不同的 early-stop 主指标制造优势；主选择键固定为 validation macro-AUPR，target weak AUROC只作同分破局。

## 5. 公平性风险

1. 当前 C6 使用 pos-weight BCE，而损失库中的普通 BCE/ASL/DB 默认缩放不同；必须保持优化器和预算一致，不额外为单个基线调参。
2. DB/CB/Logit Adjustment依赖 fit 标签统计，只能从 fit role计算。
3. 两阶段方法总训练预算不得翻倍；需将 12 epochs 固定拆为 representation 8 + classifier 4，或按等 FLOPs预算设计。
4. F3 使用冻结周期物理特征。普通 MLP 容量对照必须接收同一 C4/temporal 输入但不得使用周期特征，参数量至少匹配 5 参数只是必要条件，不足以构成强容量对照。
5. 所有方法必须共同使用 RawSaw 标签路由，否则 macro 指标比较会被 SawFM 模块差异混杂。

## 6. 当前证据边界

可以主张：F3 已经超过 C4、C4-only 弱正加权头、固定残差和 block-shuffle 同容量控制，并在多数 development-v3 分层中方向稳定。

不能主张：F3 已经超过 ASL、DB、Focal、CB、Logit Adjustment或最新多标签长尾方法。上述比较在 development-v3 尚未执行。

