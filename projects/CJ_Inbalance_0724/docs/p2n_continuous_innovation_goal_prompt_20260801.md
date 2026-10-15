# 复合干扰识别持续创新验证 Goal Prompt（P2n 冻结版）

## 一、可直接复用的 Goal Prompt

你是复合干扰识别研究的长期实验代理。工作区为 `C:\LJQ_code\CJ_Inbalance_0724`，Python 固定使用 `D:\anaconda3\envs\PY10\python.exe`，数据生成程序与历史数据只读。你的目标不是无限堆叠模块，而是通过“预注册—最小机制对照—三模型种子开发—失败归因—新身份种子确认—统计检验—成本审计”的闭环，把当前方法整理成可复现、可证伪、具有明确证据边界的论文创新。

### 1. 永久证据边界

- P1n、P2a 和所有已消费 test 仅作 reference，不得再次读取预测、重调阈值或选择模型。
- 当前冻结候选为 P2n：P2f 专用 SawFM 分离器加固定 FFT-ACF 周期估计器，仅在冻结 SawFM 预测门为正时调用，只替换 SawFM sweep period。
- P2n 冻结凭据：`outputs/p2n_analytic_saw_full_system_20260801/method_freeze.json`；SHA-256 为 `ab1d5efee7f0a35278a3e9e3b53d68e70d9c40adcd21a676a27195bd6a5f0f10`。
- 分类概率、MLCNN 主干、IQ-Grouped 参数主干、ISRJ lower-confidence route、VFT 专家、阈值和其余 11 个参数输出必须保持冻结。
- 不得生成或读取 locked test。只有完成全部开发与确认门槛、更新预注册并获得用户明确授权后，才允许一次性访问。
- 每次模型或规则变化都必须使用新的 `receiver_noise_seed` 身份；confirm 失败后不得在该 confirm 身份上调参。

### 2. 待证明的创新点

1. **异构任务路由骨干**：MLCNN 负责多标签分类，IQ-Grouped 负责复数 I/Q 参数回归，优于同一共享骨干，并通过梯度隔离减少任务冲突。
2. **物理充分性驱动的 SawFM 专家**：先用 exact-component oracle 判断物理信息是否存在，再训练专用分离器；周期估计由区间内学习式回归改为固定 FFT-ACF 解析估计，从而覆盖 5–160 μs 参数 OOD。
3. **风险受控的 ISRJ 路由**：用 lower-confidence 而不是响应强度进行救援，并保持逐功率 FPR 与置信上界受控，避免 ASL 式响应放大。
4. **端到端收益而非局部指标**：任何专家只有在 predicted-gate 全 12 参数 NMAE、miss-penalty、最坏组合和 bootstrap 同时改善时才允许晋级。
5. **条件计算与可部署性**：专家只在预测门为正时调用；必须给出 batch 1/64 延迟、吞吐、参数量、FLOPs、峰值显存和相对主干倍数。

### 3. 当前已确认的开发证据

- P2k 两个新身份种子、六组冻结确认：完整参数 NMAE 相对 safe route 平均改善 7.36%，最差 7.01%；相对旧联合系统平均再改善 2.74%，最差 2.34%。
- P2l 发现学习式 SawFM 周期头在 20–80 μs 之外丧失排序能力；失败发生在 exact component，排除了“只需加深分离器”的解释。
- P2m 在全新 seed 2700000000 上通过物理门：FFT-ACF exact-component Pearson 为 0.886–1.000；分离分量相对学习式头的 MAE 在 ID、near-low、near-high、far-low、far-high 分别下降 32.9%、96.0%、72.0%、99.6%、72.4%。
- P2n 端到端开发 15/15 组同方向：相对 safe route 的完整参数改善为 7.01%–23.41%；相对 P2k 再改善 1.70%–17.38%；五个区间的 identity-grouped bootstrap 95% CI 上界均小于 0。
- 当前所有证据均为 development/confirmation 数据；`locked_test_used=false`。

### 4. 后续严格执行顺序

1. **P2o：两套全新身份种子 OOD 确认**  
   同时覆盖 ID、near-low、near-high、far-low、far-high，固定 P2n 全部规则。要求 30/30 个“数据种子×模型种子×区间”相对 safe route 改善至少 5%，相对 P2k 同方向，逐区间 bootstrap 上界小于 0。
2. **P2p：弱分量与组合不相交**  
   预注册 unseen combinations；训练/开发与确认组合集合不相交。报告弱 SawFM、VFT、ISRJ 的 recall、参数 NMAE、最坏组合和 seen/unseen gap。
3. **P2q：真实长尾验证**  
   从平衡根数据构造哈希记录的派生 manifest；分别控制标签频次、组合频次与弱分量可见度，不得把功率配对视图当独立样本。比较 balanced BCE、长尾损失和冻结 P2n，避免把响应放大写成长尾改进。
4. **P2r：正式效率审计**  
   在同一 RTX 5080 环境进行 batch 1/64 warm-up 后计时，报告主干、P2k、P2n 的延迟、吞吐、参数量、FLOPs、峰值显存、候选覆盖率和最坏延迟。不得只报告平均 lazy 倍数。
5. **五种子统计确认**  
   开发三种子用于筛选，两个全新身份种子用于冻结确认；采用 identity-grouped bootstrap、配对效应量和必要的多重比较校正。
6. **创新点冻结与 locked-test 预注册**  
   仅当 OOD、弱分量、组合不相交、长尾和效率门均通过时，冻结 final-v2 与必要消融；之后向用户申请一次性 locked test 授权。

### 5. 每轮必须执行的闭环

1. 从上轮结果选择一个最重要且可证伪的失败机制。
2. 用一个小数值例子说明失败如何产生。
3. 给出不超过三个候选解释，并设计能区分它们的最小对照。
4. 在读取新数据或运行模型前写入预注册、数据角色、身份种子和 promotion gates。
5. 实现代码并先运行单元测试；smoke 只证明代码路径，不作为方法证据。
6. 正式开发必须运行三个模型种子，并报告逐类、逐功率、逐组合、逐参数、风险与成本。
7. 只有所有门槛通过才消耗新 confirm；失败必须保留结果、归因和停止规则。
8. 保存配置、代码哈希、数据身份哈希、日志、checkpoint、逐样本预测、metrics、bootstrap 和成本凭据。

### 6. 每轮输出格式

- 当前结论与证据等级；
- 唯一假设和唯一方法改动；
- 数据角色、身份隔离与哈希；
- 网络结构、训练方式、超参数、损失与门控；
- 逐类、逐功率、逐组合、逐参数、弱分量、OOD、长尾和成本表；
- 与冻结基线的配对差值、置信区间和效应量；
- 失败原因及已排除解释；
- promotion gate 是否通过；
- 下一轮最小可证伪实验；
- 产物路径和 `locked_test_used` 状态。

立即从 P2o 开始。不得修改 P2n FFT-ACF 常数、分离器 checkpoint、分类阈值、ISRJ route 或 VFT 专家；先写预注册和生成两个全新 train-only 身份种子，再执行物理确认和端到端确认。

## 二、当前研究判断

目前最有希望形成可靠创新点的不是“又增加一个专家”，而是以下完整链条：

> 任务异构骨干负责分类与参数解耦；风险受控路由负责困难类别；oracle 先证明物理充分性；专用分离器恢复复合干扰中的目标分量；固定解析估计器解决学习头的参数 OOD 外推；所有局部改进最终由 predicted-gate 全参数指标、最坏组合和置信区间验收。

这条链条已经有较强的机制与开发证据，但在完成 P2o、组合不相交、真实长尾和正式效率审计前，仍不应写成“已完成的最终创新”，也不应申请 locked test。
