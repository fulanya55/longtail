# Route A：RAAT 训练前停止决定与 H1j 证据缺口审计

日期：2026-08-02  
结论：**停止 RAAT-H0/v1.2 的任何训练和指标计算；不把它的 train-only 数据用于选择。将 VFT 证据转为对既有 H0i/H1j 的固定分数审计。**

## 1. 为什么在身份审计通过后仍停止

RAAT 的新 train-only 源已经按冻结协议 materialize，身份、63 组合、角色和 test-v3 seal 全部通过；但在第一次训练前的历史检索中发现了此前未纳入 Stage 0 矩阵的两条记录：

| 既有记录 | 已建立的事实 | 与 RAAT 的关系 |
|---|---|---|
| H0i | 正确已知 LFM 去调频加相邻相位增量，在新配对 VFT Doppler 干预上对 phase-permuted reference 有物理可辨识性 | RAAT-VFT 的“盲对齐后 dechirp / 残余相干”与其同属参考对齐相位残余机制 |
| H1j | 在新六标签复合源上，正确 dechirp 标量与 phase-permuted reference 的同维控制存在弱 VFT 排序增量 | 这是 RAAT-VFT 本应首先验证的同一物理假设的已有 H1 证据 |
| 历史 MFT matched-filter delay profile | 已测试参考 LFM 对齐的 delay profile 与错误 chirp 控制 | RAAT-MFT 的 delay-multiplicity 图是该已探索观测的地图化扩展，不足以单独主张为新的物理假设 |
| H0g/H0h ISRJ aligned periodicity | ISRJ 周期性已有跨参数/噪声的窄机制证据 | RAAT-ISRJ 虽不用旧特征，但单独一个新的局部图不能补足“至少两个独立新专家”的候选要求 |

因此继续训练 RAAT 会把“同一 VFT 参考去调频物理线索的地图化”当作新候选，违反本 Goal 的最大文献/既有机制距离和禁止重命名规则。这是**预训练发现的重叠**，不是 RAAT 的阴性性能结果。

## 2. RAAT 资产的严格处置

- 已 materialize 的文件：`C:\LJQ_code\干扰数据\复合干扰数据0802_raat_h0_v1\closedset_6class_train.mat`，SHA-256 `a68980badf719e63b2cc59ae566de0dd7f2f5a7ef5a5c1963b4126acdfe53a65`。
- 已通过的仅为 source identity audit：5040 唯一 receiver identities，`978677743--978682782`，63 组合、train-only 和 test-v3 seal。
- **未做**：RAAT feature extraction、网络训练、checkpoint、预测、AUROC/AUPR、阈值、校准、融合或任何 test 访问。
- 该源标记为 `quarantined_unused_due_to_pretraining_overlap`; 不删除、不重生、不拿来改用作 H1/H2；除非未来一个独立、重新 Stage-0 审计的新候选明确登记其用途，否则不能使用。

## 3. H1j 是积极线索，不是完成证明

H1j 的固定结果是 candidate-minus-control weak VFT AUROC `+0.048715`、AUPR `+0.017996`，strong AUROC `+0.219400`，而且 source identity/test sealing 审计通过。这支持继续核验 VFT 的参考去调频线索。

但它尚不能晋级为“可靠创新证据”，原因是：

1. 五个 optimizer seed 的主排序 delta 完全相同（1-D 单调读出不会改变 AUROC/AUPR），所以它们不是五个独立排名复现；
2. 已记录 CI 只对五个 seed 重抽，未对 evaluation receiver identities 作 paired hierarchical bootstrap；
3. H1j 是 source-level 单标签验证，未做目标之外 bitwise 保护、无交叉新源复现、未做安全六标签接回、无 H2/H3 和 test-v3；
4. 它的 matched-negative 规则仅冻结 other-five-label context，尚需被明确审计为固定统计对象，而不能事后改为更有利的分层。

## 4. 唯一允许的下一步

已对 H1j 的**固定、已预注册 candidate/control 分数**完成 read-only paired identity bootstrap 审计：两个 weak 端点 CI 都跨 0。因此停止 VFT H1j 机制，不更换参考、窗口、数据、训练、匹配规则、模型、阈值或校准。
