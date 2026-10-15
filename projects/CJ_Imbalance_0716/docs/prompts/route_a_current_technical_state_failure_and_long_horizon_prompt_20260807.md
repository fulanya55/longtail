# Route A 当前技术状态、失败复盘与长线研究规划 Prompt

## 使用方式

将下列 Prompt 交给 Codex。除非用户明确追加“执行实验”，本 Prompt 的默认任务仅是读取现有证据、审计、总结和规划，不得训练新模型，不得打开任何封存测试集。

```text
你是复合雷达干扰多标签识别项目的首席算法研究员、实验审计员和 SCI 证据链设计专家。请基于
E:\LJQ_code\CJ_Imbalance_0716 的当前正式代码、配置、结果、冻结件和报告，对 Route A 的实验进展、失败原因、
当前网络、训练协议和未来研究方向进行一次可复核的技术总审计，并形成可直接用于课题组讨论和论文规划的中文报告。

一、环境与不可越过的边界

1. 正式项目根目录：E:\LJQ_code\CJ_Imbalance_0716。
2. Python 环境：D:\anaconda3\envs\PY10\python.exe；MAT/HDF5 操作不得使用 base Python。
3. 正式数据来源位于 E:\LJQ_code\干扰数据；C:、D: 中的路径仅可作为历史证据，不得机械重写历史报告、manifest、receipt 或 hash。
4. Route A 与 Route B 是不同研究问题。Route A 讨论六类识别、频次长尾、标签内可见性和弱分量；Route B 的参数估计或半监督结果不得拼接为 Route A 的识别证据。
5. 必须区分：开发集、校准集、阈值集、early report、locked test、test-v2、test-v3、smoke/readiness 和 proposal。
6. 历史 locked test 与 test-v2 已经消耗，不得重新筛选、校准、调阈值或组合模型。test-v3 必须保持
   access_authorized=false、analysis_allowed=false、used=false，不得读取其 I/Q、标签、预测和指标。
7. 根 0716 数据是平衡数据。任何 label-LT、combination-LT 或 dual-axis 结论都必须来自显式派生、hash 记录的 manifest。
8. 同一 receiver_noise_seed 的不同功率视图属于同一 receiver identity，必须进入相同数据角色，不能作为独立样本跨 split。

二、证据等级

报告中的每项结论必须标注为：

- [F] confirmed/frozen：已在预注册、冻结或一次性独立测试中得到支持；
- [A] bounded/label-local：只在开发集、特定标签、特定生成器或有限控制中成立；
- [N] negative/stopped：未过门槛、被反证或按科学停止规则终止；
- [U] running/uncertain：缺少 terminal receipt 或结果尚不完整；
- [T] proposed/not validated：未来设计，不得写成既成事实。

三、必须严格分开的两个“当前系统”

1. 历史已做一次性 locked-test 的候选：RawSawCrossSeedFPR15+H2Others。
   它由旧 F2_DFMS_Full 表征、五标签 H2_PosBCE 冻结分类头、SawFM 原始 I/Q 局部前端、SawFM Platt 校准和固定跨 seed 标签路由组成。
   它是目前唯一具有正向一次性 locked-test 系统级证据的 Route A 候选，但该测试已经消耗，不能继续调参。
2. 当前继续研究的正式锚点：B0_F2_DFMS_Full。
   它在全新的 backbone-development-v1 上从随机初始化训练，用于 M1-M4 和 IP-CVE 等开发审计；它不是新的 final candidate，
   未使用 test-v3，也不能用历史 locked-test 的结果替它背书。

必须固定引用的核心结果是：历史候选在一次性 locked test 上的 macro-AUPR 为 `0.893382`，对应 H2 基线为 `0.861894`；
当前 B0 在 backbone-development-v1 上的 macro-AUPR 为 `0.843746`，MFT/VFT/ISRJ 平均 weak AUROC 为 `0.555864`、
weak recall 为 `0.237524`、strong recall 为 `0.771434`、FPR 为 `0.195215`。这些结果来自不同数据和不同系统，只能分别报告。

报告必须给出两套系统各自的用途、数据来源、参数规模、训练方式、结果和证据边界，禁止将二者混为同一 checkpoint 或同一实验。

四、当前 B0 网络必须写到可复现的层级

输入是 [B,2,3000] 的 I/Q。说明以下结构：

1. 复数编码器保持 I/Q 分离：stage1 k=7 输出 16×3000；平均池化 /2；stage2 k=5 输出 32×1500；
   stage3 复数多尺度残差输出 48×750；stage4 输出 64×375。
2. 延迟融合：Cartesian 分支将 real/imag 拼接后的 128 通道投影到 128；magnitude 分支由 64 投影到 128；
   phase 分支使用 lag=(1,4,16) 的归一化复数相位差 cos/sin，共 384 通道，经 32 隐层投影到 128，
   并以初值 0.1、tanh 有界的残差尺度接入。
3. 时序主干：128 通道的多尺度 depthwise 残差块，kernel=(3,7,15)，dilation=(1,2,4)，融合后保持 [B,128,375]。
4. 池化与表示：不使用 attention；全局 mean+std 得到 256 维；Linear(256,128)+LayerNorm+GELU+Dropout(0.2) 得到 128 维表征。
5. 分类器：Dropout(0.2)+Linear(128,6)。
6. 参数量：总计 257,511；encoder 128,416；late fusion 41,793；temporal backbone 53,376；feature projection 33,152；classifier 774。
7. 报告计算量和硬件测量时必须注明机器相关性：约 211,857,536 conv/linear MACs/sample；历史测得 batch-1 CUDA 均值延迟约 4.056 ms，峰值 CUDA memory 约 1,051,499,008 bytes。

五、训练、损失、校准和阈值协议

必须写明：

1. B0：AdamW；backbone lr=5e-5；classifier lr=5e-4；weight decay=1e-4；batch=64；max epoch=36；patience=8；
   gradient clip=5.0；seed=2039-2043；complex_rms 归一化。
2. 训练集使用显式 label-LT fit manifest，标签计数为 [1600,400,1213,919,696,528]；各 seed 的 fit/validation/early/calibration/threshold
   receiver/sample 必须互斥。
3. B0 分类损失是 asymmetric focal：gamma_pos=0、gamma_neg=3。
   元素级形式为 BCEWithLogits × [y(1-p)^0+(1-y)p^3]；总损失为五个非 SawFM 标签均值 + 0.2×SawFM 损失均值。
   明确指出：该损失会抑制容易负样本，但所有正样本的 focal 因子仍为 1，因此不区分 weak positive 与 strong positive。
4. 模型选择：validation 上 FMB/SinFM/MFT/VFT/ISRJ 的 macro-AUPR；tie-break 为 MFT/VFT/ISRJ weak-vs-negative AUROC。
5. 校准：每标签独立 Platt logistic regression，C=1e6、lbfgs、max_iter=1000，仅使用 calibration role。
6. 阈值：threshold role 上按 0.05:0.01:0.95 网格，在负样本 FPR≤0.20 约束下最大化 weak-positive recall；early role 只报告一次。
7. 历史 H2_PosBCE 分类头：冻结 F2，随机初始化 Linear(128,6)，pos_weight_k=Nneg_k/Npos_k；AdamW lr=1e-3、wd=1e-4、batch=256、
   max epoch=60、patience=12。
8. 历史 Raw SawFM 前端：raw I/Q RMS 归一化；hidden=16；kernel=7/31/63；top-k=16；pool k=4/s=4；dropout=0.1；
   88 维统计进入 Linear(88,32)-GELU-Dropout-Linear(32,1)；参数 5,521；lr=3e-4、wd=1e-3、batch=64、epoch=40、patience=8。

六、必须覆盖的实验进展与失败线

按时间和科学问题整理，不得只列“最好结果”：

1. Base-DB 相对 Base-ASL 的频次长尾基线优势，以及 DB 不能解决标签内部 weak/strong 可见性的限制。
2. PAM-IQ 的早期 I/Q 融合、共享显著性注意力和共享池化问题；F2 延迟融合和 mean/std 安全路径的改进。
3. 冻结 H2 分类头的正向证据，以及 H3 弱排序头、H4 共享 Saw 专家的失败。
4. RawSaw 对 SawFM 的独立机制证据和一次性 locked-test 系统收益，同时明确它不能自动外推到 MFT/VFT/ISRJ。
5. GradientContrast/test-v2 的正向结果与后续 specificity/equal-capacity 审计失败；不得宣称统一 dual-axis 方法已成立。
6. ISRJ aligned periodicity 的标签局部收益与统一 OOD 候选失败。
7. MFT、VFT 固定物理统计可观测，但学习型增量未过 equal-capacity、安全或准入门槛的“observability-to-learning gap”。
8. backbone-development-v1 的 M1 anti-alias、M2 cross-stage pyramid、M3 long-receptive-field evidence block、M4 max/top-k local statistics 均未过预设弱指标门槛。
9. IP-CVE Stage 0 数据合同通过，但 P0 冻结 B0 可辨识性未战胜控制；P1 被禁止执行；单通道本地机制族按预注册结论关闭。

七、失败原因分析框架

必须形成“现象 -> 直接证据 -> 根因判断 -> 已排除替代解释 -> 尚未排除解释”的证据链，至少覆盖：

1. 单通道混合观测下的弱分量可辨识性；
2. component existence 与 label attribution 的区别；
3. BCE/Focal/DB 的监督目标与弱证据定位不匹配；
4. 强正样本支配共享表征和分类边界；
5. mean/std 稀释稀疏证据，max/top-k 又被强分量和 nuisance 支配；
6. 标签特定结构与共享梯度的负迁移；
7. 固定物理统计可分与端到端模型增益之间的断层；
8. 校准/阈值只能放大已有排序，不能修复 AUROC 级不可分；
9. 阈值在 receiver、背景和功率分布间的运输风险；
10. 合成数据、单通道观测和缺少 real/HIL 外部验证造成的外推限制；
11. 已消费测试集造成的证据治理约束；
12. 历史 tested system 与当前 research anchor 版本歧义。

八、创新点审计

分别审计下列主张，给出“现有支持、反证、缺失控制、允许表述、禁止表述、下一准入门槛”：

1. 频次长尾与标签内可见性退化是两个独立轴；
2. RawSaw 是 SawFM 标签特定弱证据前端；
3. ISRJ aligned periodicity 是窄范围物理组件；
4. 统一 VTSE/VBF 或通用可见性增强网络；
5. stronger backbone 能解决弱分量召回；
6. IP-CVE 单通道反事实表征；
7. 未来多观测/多通道边际证据网络。

九、改进方向和准入顺序

改进必须按风险和信息增益排序：

A. 首选方向 [T]：引入可部署、非 oracle 的第二物理观测，例如同步主通道+校准辅助通道、两种固定接收机设置或 real/semi-physical/HIL 数据。
   必须先做零训练 Stage -1/H0 可辨识性试验；只有 matched observation 明显战胜 channel-swap、transfer perturbation、time permutation、unmatched identity
   和 equal-capacity controls，才能训练网络。
B. 若 H0 通过，再设计多观测网络 [T]：主/参考复数编码器、跨通道归一化 coherence、magnitude ratio、transfer-corrected residual、lagged phase difference；
   保留 B0 global safety path；为 MFT/VFT/ISRJ 使用标签特定证据头；接回残差绝对值不超过 0.2，gate 零初始化。
C. 若无法取得新观测，则缩小论文主张：保留 RawSaw 的 SawFM locked-test 证据、ISRJ aligned-periodicity 的窄机制结果、MFT/VFT 物理可观测边界，
   将 M1-M4 和 IP-CVE 写成严谨的负结果/方法边界；不得再宣称统一双轴网络已经建立。
D. 只有获得有效 visibility mechanism 后，才运行 frequency-balanced/label-LT × high/low-visibility 的 2×2 双轴验证，并使用新的、hash 记录的 identity-isolated development/confirmation 数据。

禁止把下列内容作为下一轮主线：继续扩宽通用 backbone、堆叠注意力、重新组合 M1-M4、无机制的 loss-weight sweep、阈值救援、复用已消费数据源或测试集、
以及推理时使用精确模拟 absent-view 的 oracle 第二视图。

十、要求输出的报告结构

1. 执行摘要和当前总裁决；
2. 研究范围与证据边界；
3. 实验进展时间线与状态表；
4. 历史 locked-test 候选的完整结构、参数、训练和结果；
5. 当前 B0 research anchor 的逐层网络结构；
6. 当前超参数、损失、模型选择、校准和阈值；
7. 各路线失败复盘和统一根因图；
8. “特征是否充分学习”与“backbone 是否仍需增强”的结论；
9. 当前创新点证据审计矩阵；
10. 技术难点优先级；
11. 未来 0-3、3-6、6-12 个月长线计划，每阶段给出数据、模型、控制、指标、硬门槛和停止规则；
12. 允许做与禁止做的事项；
13. 可复现性和待保全产物清单；
14. 一句话的当前最强论文表述，以及一句话的禁止过度表述。

报告必须优先使用仓库中的正式 JSON、CSV、manifest、receipt、frozen snapshot 和结果报告。指标必须说明数据角色和证据等级。
若证据不足，写“未知/未验证”，不能猜测。提案统一标注 [T]。默认不执行任何新实验。
```
