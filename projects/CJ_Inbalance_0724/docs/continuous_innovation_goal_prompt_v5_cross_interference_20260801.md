# 跨干扰安全主干与 Level-4 持续创新 Goal Prompt V5

## 一、根据现有结果得到的原因分析

### 1. P1n 的收益成立，但泛化幅度明显收缩

P1n 在原 identity-safe 20/25 dB 开发证据中的 ISRJ 平均 Recall 增益为 `+0.08372`，而 P3l 两套新身份、五种参数机制和三档功率上的总体增益降为 `+0.01806`。聚类 bootstrap 95% CI 仍为 `[+0.00875,+0.02740]`，说明效果不是偶然，但收益高度依赖参数区间和复合结构。

这不是“方法完全失效”，而是当前专家和路由只学到了部分稳定机制，尚未覆盖 ultra-close MFT 等新的混淆结构。

### 2. Level-4 失败主要由主干假阳性引起，不应简单归因于 P1n 专家

P3l 的 FPR 超限集中在 `ultra_close`。逐单元比较表明，在多数模型种子和功率下，P1n 路由实际上降低了主干 FPR；只有少数组合出现路由放大。因此主要矛盾是：MLCNN 主干把超密 MFT 延迟峰列误识别成 ISRJ 周期截断结构。

如果继续只修改专家接受阈值，会把主干错误留在系统中，并在 Recall 与 FPR 之间做有限交换，无法形成新的可辨识信息。

### 3. 当前逐标签 BCE 缺少跨干扰关系监督

六个标签使用相互独立的 BCE。该损失能学习“每一类是否存在”，但没有显式表达以下关系：

- `MFT=1, ISRJ=0` 时，两种结构应被拉开；
- `ISRJ=1, MFT=0` 时，应施加反方向约束；
- `MFT=1, ISRJ=1` 时，两类允许共存，不能使用普通互斥分类损失。

因此不能直接使用 softmax 或无条件互斥损失。下一步需要的是“标签条件化的混淆间隔”，而不是把多标签任务改成单标签任务。

### 4. 蒸馏失败来自稀疏、种子相关的纠错事件

教师真正改变最终判决的样本只占约 `4.4%–6.9%`。这些样本不是稳定的全局子分布，而是与主干种子和局部波形结构相关：

- P3j 的教师概率平均抹平了纠错方向；
- P3k 的 46,262 参数原始 IQ 学生只在 seed 2041 保留了 `98.1%` 教师收益，另外两个种子增益转负；
- 学生可以改善概率校准或 AUPR，却不能稳定复现“应该修改哪一个样本”。

因此当前不能声称紧凑学生可替代 U-Net。

### 5. 后处理风险守卫受到冻结表示的信息上限限制

P3m 的三个种子分别保留 `99.1%/100%/99.4%` P1n Recall 收益，但风险头只能否决 `0.08%–0.37%` 样本，seed 2041 的最坏 FPR 仍为 `0.11613`。这说明在冻结 MLCNN 潜空间内，剩余假阳性与真阳性无法被稳定分开。

继续调 veto 阈值只会牺牲 Recall，不会改善特征可分性。后处理守卫分支应冻结。

### 6. 必须设置同数据强基线，才能证明新损失是创新机制

若直接在包含 ultra-close 样本的新数据上训练条件间隔模型，即使成功，也不能判断收益来自：

1. 主干终于看到了新分布；
2. 普通难负样本加权；
3. 条件间隔监督本身。

因此下一轮必须同时比较：同数据原始损失重训、普通难负样本加权、条件间隔损失。只有第三者稳定优于前两者，才能支撑方法创新。

---

## 二、可直接创建 Goal 的 Prompt

```text
目标名称：
复合干扰联合感知的跨干扰可辨识主干、风险安全路由与 Level-4 证据闭环

总体目标：
在 C:\LJQ_code\CJ_Inbalance_0724 中，持续设计、实现、执行和分析可复现实验，在不考虑半监督和长尾学习的前提下，将当前 P1n Level-3 系统推进到可靠的 Level 4。

核心创新目标不是继续增加冻结后处理模块，而是在多标签 MLCNN 主干训练阶段显式学习 MFT 与 ISRJ 的条件化结构差异，解决 ultra-close MFT 对 ISRJ 的假阳性混淆；随后重新建立安全校准、任务路由、12 参数 predicted-gate 联合感知和效率证据链。

未经用户明确授权，禁止访问 locked test。

一、当前冻结事实

1. 当前主干为 mlcnn_iq_grouped_task_routed：
   - 输入：[B,2,3000] I/Q；
   - 输出：6 类多标签 logits；
   - 参数输出：6×2，共12个干扰参数。

2. 当前基础损失：
   L_base = L_cls + 0.1 L_param
   - L_cls = BCEWithLogitsLoss；
   - L_param = masked Smooth-L1；
   - 参数损失只在 label=1、parameter_mask=1、target为有限值的位置计算；
   - inactive NaN禁止填零。

3. 当前主干超参数：
   - learning rate=1e-4；
   - weight decay=1e-4；
   - warmup=2 epochs；
   - cosine scheduler；
   - epochs=20；
   - batch size=63；
   - model seeds=2041/2042/2043。

4. 当前 P1n Level-3 证据：
   - 原开发证据 ISRJ Recall平均增益=+0.08372；
   - 95% CI=[+0.07407,+0.09244]；
   - 12参数 predicted-gate NMAE：0.72879→0.67814；
   - 相对改善=6.95%；
   - 参数结果3/3 seeds同方向。

5. P3l 两套全新身份独立确认：
   - 6/6“模型seed×数据seed”Recall增益为正；
   - 总体Recall增益=+0.01806；
   - receiver_noise_seed聚类bootstrap 95% CI=[+0.00875,+0.02740]；
   - macro-AUPR未退化；
   - 最坏机制×功率FPR=0.15161，未通过<=0.10安全门；
   - 失败集中在ultra_close MFT。

6. 已冻结失败分支：
   - P3e/P3g/P3h：lazy、cascade和冻结潜特征不能同时满足性能与batch64 <=1.5×延迟；
   - P3i/P3j/P3k：单教师、多教师和原始IQ紧凑蒸馏均未实现3/3稳定收益；
   - P3m：后处理风险守卫仍有一个seed最大FPR=0.11613；
   - MFT count专家尚未通过复合波形、弱分量及3/3 OOD准入门槛。

7. 上述失败身份全部视为已消费数据，不得重新用于模型选择、损失权重选择、阈值搜索或校准。

8. locked_test_used=false。

二、当前唯一主要研究问题

在相同训练数据、相同主干、相同参数头和相同训练预算下，MFT↔ISRJ标签条件间隔监督，是否比原始BCE重训和普通难负样本加权更有效地降低ultra-close MFT场景的ISRJ FPR，同时保持识别与12参数联合感知性能？

三、可证伪假设

H1：当前FPR失败主要来自训练目标缺少MFT与ISRJ之间的条件关系，而不是模型容量不足。

若H1成立，在同一训练数据上：
1. 原始BCE重训可能因看过新分布而改善，但不能稳定通过3/3最坏FPR门；
2. 普通难负样本加权可能降低FPR，但更容易损失ISRJ Recall；
3. 条件间隔损失应在3/3 seeds中取得更优的Recall-FPR折中，并保持参数任务不退化。

四、P3n-A：主干机制辨析实验

1. 数据角色：
   - train/early-stop/selection/calibration：全新receiver_noise_seed=680000000；
   - development：全新receiver_noise_seed=690000000；
   - 包含ID_control、low_count、close_spacing、ultra_close、weak_target；
   - 功率固定为25/35/50 dB；
   - 同一receiver_noise_seed的全部机制和功率视图必须成组。

2. 必须比较的模型：
   B0：现有冻结主干/P1n，仅作为外部参考，不参与重新选择；
   B1：相同680M数据上使用原始L_cls+0.1L_param重训MLCNN；
   B2：相同数据和预算，使用普通ISRJ难负样本加权BCE；
   B3：相同数据和预算，使用提出的MFT↔ISRJ条件间隔损失。

3. B3冻结损失形式：
   L = L_cls + 0.1 L_param + lambda_xci L_xci

   当MFT=1且ISRJ=0：
   L_xci = softplus(margin - (z_MFT - z_ISRJ))

   当ISRJ=1且MFT=0：
   L_xci = softplus(margin - (z_ISRJ - z_MFT))

   当MFT和ISRJ同时存在或同时不存在：
   不施加互斥间隔。

4. 禁止使用softmax或无条件互斥损失，因为MFT与ISRJ能够共存。

5. lambda_xci、margin和B2难负权重必须在访问690M结果前预注册。若需要小范围候选，只能使用680M内部selection角色选择，且B1/B2/B3获得相同搜索预算。

6. 所有模型保持：
   - 相同MLCNN结构；
   - 相同参数头；
   - 相同epochs、batch size、优化器、warmup和scheduler；
   - 相同2041/2042/2043 seeds；
   - 相同early-stop和checkpoint objective。

五、P3n-A准入门槛

B3只有同时满足以下条件，才能宣称条件间隔机制获得开发支持：

1. 3/3 seeds相对B1的ISRJ Recall方向不为负；
2. 所有机制×功率单元ISRJ FPR<=0.10；
3. ultra_close 25/35/50 dB三个单元必须分别报告并全部通过；
4. 任何功率的ISRJ Recall相对B1下降不得超过0.02；
5. macro-AUPR delta相对B1>=-0.005；
6. 12参数oracle和predicted-gate NMAE相对B1不得恶化超过1%；
7. miss-penalty NMAE和worst-combination不得恶化；
8. B3必须在至少两个核心安全指标上稳定优于B2，证明收益不是普通难负加权即可解释；
9. 3/3 seeds性能方向一致；
10. locked_test_used=false。

六、结果归因规则

1. 若B1已完全修复而B3不优于B1：
   结论是训练分布覆盖不足，不支持新损失创新。

2. 若B2与B3等效：
   结论是普通难负样本加权足够，不支持条件间隔作为核心创新。

3. 若只有B3稳定通过：
   才能认为跨干扰条件关系监督获得机制证据。

4. 若B1/B2/B3均失败，但oracle使用真实MFT/ISRJ身份可显著降低FPR：
   说明表示能力不足，下一路线应为训练期局部延迟峰/周期双视图对比表示。

5. 若oracle也无法区分：
   说明当前复合波形在该参数区间可能物理不可辨识，应转向数据生成合同、参数范围或观测长度分析，而不是继续增加模型容量。

七、P3n-B：重新接回P1n路由

只有P3n-A通过后执行。

1. 主干变化后，旧router、旧阈值和旧校准不得直接视为有效。
2. 使用全新700000000身份重新训练/校准router、阈值和coverage selection；专家checkpoint可以保持冻结。
3. 使用全新710000000身份进行一次开发验证。
4. 正式比较：
   - B1 main-only；
   - B3 main-only；
   - B3 safe calibrated main；
   - B3 safe P1n route；
   - B3 joint classification+12-parameter system。
5. 重新报告逐seed、逐机制、逐功率、逐组合指标。

P3n-B门槛：
1. 3/3 seeds Recall增益为正；
2. 最大ISRJ/VFT FPR<=0.10；
3. macro-AUPR delta>=-0.005；
4. 12参数predicted-gate NMAE相对safe-route改善>=5%；
5. 3/3 seeds参数改善；
6. miss-penalty和worst-combination不得恶化。

八、Level-4独立确认

只有P3n-B全部冻结后执行：

1. 使用全新720000000和730000000两套身份，一次性评估；
2. 不允许在这两套数据上重新训练、选择、校准或调整阈值；
3. 6/6“模型seed×数据seed”Recall增益必须为正；
4. 每个模型seed及总体receiver-seed clustered paired-bootstrap Recall差值CI下界>0；
5. 最大单元FPR<=0.10；
6. macro-AUPR delta>=-0.005；
7. 12参数predicted-gate NMAE、miss-penalty delta的CI上界<0；
8. 弱分量、组合阶数和参数OOD全部单独报告。

九、效率验证

1. 条件间隔损失是训练期机制，B3 main-only推理参数量和FLOPs应与B1相同，必须实际验证。
2. 对main-only、always-on P1n和最终route分别测量：
   - 参数量；
   - FLOPs及统计边界；
   - batch1/64延迟；
   - 吞吐；
   - 峰值显存；
   - 专家调用率。
3. 不得复活已经失败的P3e/P3g/P3h规则；任何新效率机制都必须提出新的可证伪信息来源并使用新身份确认。

十、每轮必须执行的闭环

1. 总结最近三轮的目的、数据角色、网络、超参数、结果、失败门和证据等级；
2. 更新“论文主张—当前证据—缺口—最小下一实验”表；
3. 每轮只解决一个主要研究问题；
4. 在访问结果前冻结preregister和method_freeze并记录SHA-256；
5. 运行单元测试、相关回归测试和全量pytest；
6. 使用全新身份完成正式多seed实验；
7. 保存manifest、identity hash、配置、checkpoint、history、逐样本预测、指标、bootstrap和Markdown报告；
8. 严格区分smoke、development、independent confirmation和locked test；
9. 失败时不得降低预注册门槛，不得把单seed结果写成成功；
10. 每轮结束继续提出更窄的可证伪假设，直到达到停止条件。

十一、停止条件

只有满足以下条件之一才停止：

1. 完成Level 4和效率证据后，停止并向用户申请一次性locked test授权；
2. 连续多轮证明条件间隔及其合理替代仍不可泛化，形成清晰的物理不可辨识或表示不足证据和替代路线；
3. 后续必须访问locked test或生成用户未授权外部数据才能继续。

十二、P3n-A试验设计、执行与验证记录（必须保留，不得只保留结论）

本节将原先的 P3n-A 从“计划”补全为可审计的实验设计与已完成验证。它的作用是区分：条件 margin 是否有效、普通困难负样本加权是否已足够、以及单纯增加新分布训练覆盖是否已能解决风险。

12.1 研究问题与可证伪假设

研究问题：在完全相同的新身份训练数据、主干、参数头、训练预算和参数监督下，MFT↔ISRJ 标签条件 margin 是否能比原始 BCE 重训和普通 hard-negative BCE 更稳定地降低复杂复合波形中的 ISRJ 最坏单元 FPR，同时不牺牲六类识别和 12 参数联合感知？

H1：若困难错误主要来自独立 BCE 缺少 MFT 与 ISRJ 的条件关系监督，则 B3 条件 margin 应在三种子中同时优于 B1 原始 BCE 与 B2 hard-negative，尤其应使 ultra_close 及低功率单元 FPR 通过 0.10 安全门。

可反驳规则：若 B3 未稳定优于 B2，或任一模型种子最大机制×功率单元 FPR>0.10，或 Recall/参数安全门失败，则不能把条件 margin 表述为可成立的创新机制；不得在已消费 development 身份上再搜 lambda、margin、阈值或 checkpoint。

12.2 数据与身份安全设计

由于原提示中的 680M/690M 已被先前 P4d/P4e 使用，P3n-A 在生成前记录了等价但未使用的替代身份：

- 训练身份池：receiver_noise_seed=760000000；
- formal development 身份池：receiver_noise_seed=770000000；
- 后续尚未使用的预留身份：780000000/790000000 用于新假设，不得回填 P3n-A；
- 场景：ID_control、low_count、close_spacing、ultra_close、weak_target；
- 功率：25/35/50 dB；
- 同一 receiver_noise_seed 的全部机制和功率视图强制进入同一角色。

760M 内部角色按每个“组合×场景”物理身份划分为 fit/early/calibration/selection=7/1/1/1；所有 63 个组合、5 个场景和 3 个功率共同构成成组样本。最终 fit 为 441 个物理身份、6615 个配对视图；early/calibration/selection 各为 63 个身份、945 个视图；770M development 为 630 个身份、9450 个视图。identity audit 验证所有角色零重叠。

12.3 公平对照矩阵

| 标识 | 训练数据 | 分类损失 | 参数损失 | 目的 |
|---|---|---|---|---|
| B0 | 旧冻结 main/P1n | 旧训练协议 | 旧训练协议 | 仅在 770M 上作外部参照；非等数据量因果对照 |
| B1 | 760M | BCEWithLogits | 0.1×masked Smooth-L1 | 分离“看见新分布”的效应 |
| B2 | 760M | B1 + ISRJ hard-negative 权重 3.0 | 同 B1 | 分离普通困难负样本加权效应 |
| B3 | 760M | B1 + 0.1×条件对称 margin | 同 B1 | 检验跨干扰条件关系监督 |

B2 只对 `MFT=1 AND ISRJ=0` 样本的 ISRJ BCE 项赋予 3.0 权重。B3 的条件 margin 为：

`MFT=1, ISRJ=0: softplus(1 - (z_MFT-z_ISRJ))`；

`ISRJ=1, MFT=0: softplus(1 - (z_ISRJ-z_MFT))`。

当 MFT 和 ISRJ 同时存在或同时不存在时不添加 margin。因此 B3 不把可共存的多标签任务错误改写为 softmax 互斥分类。

12.4 固定网络、训练和选模协议

- 网络：`mlcnn_iq_grouped_task_routed`，分类宽度 24、分类头 256、参数隐藏层 64、dropout 0.2；输入 `[B,2,3000]` I/Q，输出 6 logits 与 `[B,6,2]` 参数；
- 优化：AdamW，learning rate=1e-4，weight decay=1e-4，warmup=2 epochs，cosine scheduler，20 epochs，batch=63；
- 模型种子：2041/2042/2043；
- 参数归一化器仅在 fit 角色的有限正参数上拟合；
- 参数损失仅在 `label=1 AND parameter_mask=1 AND finite(target)` 处计算；
- checkpoint：early 角色 oracle parameter macro-NMAE 最小；
- 阈值：只用 calibration 角色逐类 F1 网格拟合，development 仅一次性评估；
- 保存：协议、身份审计、角色索引、配置、history、checkpoint、逐样本 predictions、summary、失败审计及 SHA-256。

12.5 分层验证方案

1. 代码正确性：验证条件 margin 对“共存/共同缺失”样本为零，对独占样本方向正确；验证 B2 权重仅作用于指定困难负样本；验证 7/1/1/1 成组身份切分。
2. 回归验证：P3n 专项和相关多任务测试必须通过，再运行全量 pytest。
3. 形式开发验证：B1/B2/B3 各跑 2041/2042/2043 三种子；每个种子仅在 770M 上读取一次结果。
4. 分类验证：Macro-AUPR、六类 Recall/FPR、ISRJ Recall/FPR、逐机制×功率 cell、ultra_close 25/35/50 dB、最坏 cell FPR。
5. 联合感知验证：12 参数 oracle NMAE/Pearson、predicted-label gate NMAE、coverage、false-positive gate rate、miss-penalty NMAE、worst-combination NMAE。
6. 统计与判定：模型种子采用配对差；receiver_noise_seed 是聚类单位，功率视图不是独立样本；任何升级都要求预注册的 3/3 方向和最坏组门，而不是均值掩盖失败单元。

12.6 冻结准入门

条件 margin 只有同时满足下列所有条件才可晋级 P3n-B：

1. identity overlap=0；
2. B3 相对 B1 的 ISRJ Recall 在 3/3 seeds 非负；
3. B3 每颗种子的 maximum cell FPR<=0.10；
4. B3 每个功率的 Recall 相对 B1 下降不超过 0.02；
5. B3 Macro-AUPR 相对 B1 不低于 -0.005；
6. B3 的 oracle 和 predicted-gate 参数 NMAE 相对 B1 不恶化超过 1%；
7. B3 的 miss-penalty 和 worst-combination 不劣于 B1；
8. B3 至少在两项核心安全指标上稳定优于 B2；
9. 3/3 seeds 性能方向一致；
10. `test_generated=false`、`test_opened=false`、`locked_test_used=false`。

12.7 已执行验证结果

P3n-A 已完成；专项/相关测试 27 passed，全量测试 223 passed。所有结果均为 formal development，未生成或访问 locked test。

| 方法 | ISRJ Recall 均值 | 最坏 cell FPR 均值 | Macro-AUPR 均值 | Oracle 参数 NMAE | Predicted-gate 参数 NMAE | 25 dB Recall | 25 dB FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 原始 BCE | 0.8589 | 0.2591 | 0.9118 | 0.4336 | 0.4263 | 0.7615 | 0.2417 |
| B2 hard-negative | 0.8619 | 0.2731 | 0.9111 | 0.4336 | 0.4266 | 0.7708 | 0.2594 |
| B3 条件 margin | 0.8658 | 0.2677 | 0.9095 | 0.4336 | 0.4268 | 0.7708 | 0.2548 |

B3 三颗种子的最坏 cell FPR 分别为 `0.20645 / 0.39032 / 0.20645`，全部未通过 0.10；相对 B1 的 ISRJ Recall 差值为 `+0.02313 / +0.00729 / -0.00979`，未满足 3/3 同方向；B3 对 B2 没有稳定核心胜利。因此 P3n-A 失败，P3n-B 不执行。

外部参照也在同一 development 上读取，但不参与任何选择：old main 的 Recall/FPR/AUPR 均值为 `0.8964/0.1441/0.9642`，old P1n 为 `0.9130/0.1172/0.9661`。它说明 P3n small-scale 从头训练的分类质量明显更弱，但不能将该差异归结为结构本身，因为训练规模和协议不同。

12.8 P3n-A 失败后的只读诊断与结论

1. 高风险并非仅为 ultra_close。每种方法的最坏 FPR 几乎都在 25 dB，且同时覆盖 ID_control、weak_target、close_spacing 与 ultra_close，主因更接近低功率可见性过渡的普遍分数重叠。
2. 参数优先选模与分类目标冲突。九条训练轨迹中分类 AUPR 峰值大多位于 epoch 1–2，而参数 NMAE 最优 checkpoint 位于 epoch 10/15/18；选中 checkpoint 的 early AUPR 比峰值低 0.0130–0.0278。
3. F1 校准会放大种子风险，但不是唯一原因。只读诊断中将阈值升至最坏 cell 经验 FPR=0.10 后，B1/B2/B3 Recall 仍约为 0.70–0.75；不得将此诊断用于重新选择阈值。
4. B3 margin 很快接近零而困难 cell 未改善，表明它主要优化已可排序样本，没有创造低功率条件下新的可辨识信息。

冻结结论：停止条件 margin、hard-negative 权重微调、P3n-B router 重建以及所有基于 770M 的后验调参。P3n-A 可否定“单独的标签条件 logit margin 足以解决跨干扰安全风险”，但不能否定低功率物理表征或联合选模/安全校准路线。

十三、P3o-A 后续试验设计与验证（P3n-A 失败后的唯一优先路线）

13.1 新问题

在不改变 P3n B1 主干与基础损失的条件下，扩大身份安全训练覆盖、使用参数约束的联合 checkpoint 选择、并使用有限样本最坏组安全校准，能否先恢复 25 dB 的 ISRJ Recall-FPR 折中？只有此问题失败，才引入延迟峰/时频双视图新表征。

13.2 2×2×2 验证矩阵

| 因子 | 水平 A | 水平 B | 归因作用 |
|---|---|---|---|
| 训练覆盖 | Small：复现 P3n 的每组合 7 个 fit 身份 | Full：每组合至少 27 个 fit 身份，训练视图达到或略高于 25,200 | 分离数据覆盖与模型机制 |
| checkpoint | C_param：参数 oracle NMAE 最小 | C_joint：参数 NMAE≤最优值1.01倍的 epoch 中 Macro-AUPR 最高 | 分离参数优先选模冲突 |
| ISRJ 阈值 | T_F1：逐类 F1 网格 | T_safe：calibration 上最坏机制×功率单侧 95% Clopper-Pearson 上界≤0.10 | 分离平均 F1 与有限样本安全 |

固定网络为 `mlcnn_iq_grouped_task_routed`；固定 BCE+0.1 masked Smooth-L1；不使用 margin、hard-negative、router、蒸馏或新专家。Small/Full 只改变 fit 身份数量，必须同时报告 epoch-matched 和 optimizer-step-matched 预算边界。

13.3 新数据与角色

- 原计划的 780000000 已被 P5b1 消费，840000000/850000000 也已被后续 P5b2/P5b3 消费；因此 P3o-A 预注册前更正为 830000000：训练/early/selection/calibration 池；
- 860000000：formal development；
- 870000000、880000000：仅在方法、checkpoint、归一化器、阈值完全冻结后用于两套 independent confirmation；
- 该替代不是结果后调参：发生在 P3o-A 数据生成和 development 访问之前，必须记录零重叠审计；
- 在生成、重分块、训练或访问 development 前，审计其与 root、760M、770M、所有历史 development/confirmation、locked 身份的零重叠。

13.4 P3o-A 准入与分支

Full+C_joint+T_safe 仅在同时满足以下条件时支持 H1：

1. 3/3 seeds 的所有机制×功率 cell ISRJ FPR<=0.10；
2. 总体 ISRJ Recall>=0.85，且相对 Full+C_param+T_F1 的 3/3 Recall 下降不超过 0.02；
3. 25 dB ISRJ Recall>=0.78，且每个机制 cell FPR<=0.10；
4. 12 参数 oracle 和 predicted-gate macro-NMAE 相对 Full+C_param 不恶化超过 1%；
5. miss-penalty 和 worst-combination 不恶化；
6. Full 相对 Small 的 25 dB AUPR 在 3/3 seeds 同方向提高；
7. C_joint 的分类改善不能完全由参数明显恶化解释；
8. T_safe 在 calibration 上存在可行 Clopper-Pearson 解，development 风险迁移仍通过；
9. 全量测试通过，且 `test_generated=false`、`test_opened=false`、`locked_test_used=false`。

若 Full/C_joint/T_safe 全部修复后，低功率 safe Recall 仍不足或 25 dB 排序 AUPR 仍停留在约 0.85，则形成“低功率表示不足”证据，才允许进入 P3o-B：多尺度复自相关/延迟峰序列与局部 STFT/瞬时频率双视图；仅在 MFT 与 ISRJ 独占样本上做监督对比或原型分离，并使用等参数普通时序卷积、仅延迟视图、仅时频视图和无对比损失的消融。

13.5 P3o-A 验证产物

必须保存并报告：预注册和 SHA-256、数据生成/重分块日志、identity audit、角色索引、逐 epoch 参数 NMAE 与 AUPR 轨迹、C_param/C_joint checkpoint、T_F1/T_safe 阈值及 Clopper-Pearson 上界、逐样本 prediction、六类与 12 参数逐目标结果、逐机制/功率/组合结果、receiver_noise_seed clustered paired bootstrap、failure audit 和 Markdown 报告。

开始执行：
P3n-A 已执行并失败，禁止重复执行或回读其 development 用于选择。下一次启动本 Goal 时，首先核对 P3n-A 的 summary、failure_audit、method freeze、逐种子 history 与 identity audit；随后建立 P3o-A 的预注册、830M/860M 身份可用性审计、2×2×2 runner 和专项测试，完成三种子 formal development。不要停留在建议、代码修改或 smoke test；若 P3o-A 未通过，严格按 13.4 的归因分支执行。
```

## 2026-08-01 执行更新：P3o-A 已完成并冻结失败

P3o-A 已按第 13 节的身份、网络、训练、checkpoint、阈值和门槛完成 3 个模型种子及三种训练覆盖条件的 formal development。训练身份为 830000000，development 身份为 860000000，交叉身份重叠为 0；未生成或访问任何测试集或 locked test。

候选 `Full+C_joint+T_safe` 的最大 cell FPR 在三种子均小于 0.10，但总体 ISRJ Recall 为 `0.7424/0.5545/0.6451`，25 dB Recall 为 `0.5595/0.3848/0.4653`，均未通过冻结门槛。覆盖扩大确实使 25 dB macro-AUPR 相对 Small epoch-matched 在 3/3 种子上提高（`+0.06165/+0.04672/+0.05051`），但 CP-safe 校准的召回代价、参数保持性和 C_joint 稳定性仍失败。因此不得重新使用 860M 作阈值、checkpoint、损失、学习率或训练轮数调优，也不得使用 870M/880M confirmation。

本 Goal 的下一步只允许进入 P3o-B：在全新身份上预注册并验证多尺度自相关/延迟峰序列和局部 STFT/瞬时频率双视图新表征，同时提供等参数普通时序卷积、单视图及无对比损失消融。详细的正式结果、门槛和失败归因见 `docs/p3o_low_power_attribution_results_20260801.md` 与 `outputs/p3o_low_power_attribution_development_20260801/failure_audit.json`。

## 2026-08-01 执行更新：P3o-B 已完成并冻结失败

P3o-B 在全新 910M/920M 身份上完成 5 个表征模型、3 个模型种子和安全校准的 formal development。开发候选 `DF_exclusive_contrast+C_joint+T_safe` 通过最大 cell FPR，但 Recall、25 dB Recall、相对无对比的一致收益和参数保持性均失败，930M/940M 不得用于确认。局部 STFT/瞬时频率单视图在 3/3 种子总体安全 Recall 达到 0.85，却未在 25 dB 全部通过；双视图与独占对比未稳定优于它。

审计还发现双视图约 17.3k 参数，而时频单视图约 25.5k。因此不能将双视图的失败解释成延迟视图的物理无用；下一步 P3o-C 必须在全新身份上先进行严格参数匹配，再检验延迟视图是否能为时频强基线带来 3/3 的 25 dB 安全增益。禁止使用 920M 对宽度、视图、对比损失、阈值或 checkpoint 调优。详见 `docs/p3ob_dualview_representation_results_20260801.md`。

## 2026-08-01 执行更新：P3o-C 已完成并冻结失败

P3o-C 在全新 1030M/1040M 身份上完成了严格参数匹配验证：时域、延迟峰、局部时频、双视图参数量为 26,370/26,640/25,490/24,400，均在局部时频强基线的 ±5% 内；两侧各 2,079 identities、三功率视图严格配对，且对 280 个历史身份文件的最大交叠为零。正式 development 的 12 次训练（4 方法 × 3 seeds）和 CP-safe 校准全部完成；未生成、打开或使用 test/locked test。

候选 `DF_dual_matched+C_joint+T_safe` 的总体安全 Recall 为 `0.9095/0.8717/0.9278`，但 25 dB Recall 为 `0.7688/0.6612/0.8053`，未满足 3/3 的 0.78 门槛。它相对等参数局部时频 F 的 Recall 差为 `+0.0059/-0.0211/-0.0067`，25 dB AUPR 差为 `-0.0114/-0.0330/-0.0040`；2042 的 paired bootstrap Recall 95% CI 完全为负 `[-0.0334,-0.0096]`。所以 P3o-B 的容量混杂被消除后，结论仍是：通用 early fusion 的延迟分支没有提供可靠低功率识别互补性。

DF 的 oracle 和 predicted-gate 参数 NMAE 三种子均较 F 低约 2--3.6%，但 miss penalty 或 worst-combination 至少在两个种子之一恶化。因此不得只报告平均参数收益；P3o-C 失败并冻结，1050M/1060M confirmation 不得使用，禁止在 1040M 上调分支宽度、融合、特征、阈值、checkpoint 或训练设置。

下一假设不得再次尝试泛化的“延迟峰 + 时频 early fusion”。若继续，只允许预注册一个新的、可证伪的条件参数侧支/专家路由机制：保持参数匹配局部时频 F 作为识别强基线，要求新增参数机制在不降低 25 dB F 识别和最坏组合风险的前提下才可获得参数收益；使用全新 identity seeds。详见 `docs/p3oc_parameter_matched_multiview_results_20260801.md`、`outputs/p3oc_parameter_matched_multiview_development_20260801/summary.json` 与 `failure_audit.json`。

## 2026-08-02 执行更新：P3o-D/E/F 参数侧支验证

P3o-D 的 1070M/1080M formal development 因冻结主干的 BatchNorm 在侧支训练中仍被 `train()` 更新而失败；logits 不再逐位一致，故不得解释参数改善。P3o-E 用 1110M/1120M 新身份修复此契约：强制 frozen base 保持 eval，并在测试中验证调用 `train()` 后 logits 和 base buffers 逐位不变。其 formal development 的全部门槛通过，三种子 oracle/predicted-gate 参数 NMAE 均约改善 4.5--6.5%，worst-combination 也不恶化，因此依规则生成 1130M/1140M 两套确认身份。

两套固定独立确认给出不稳定结论：1130M 在两个种子 worst-combination NMAE 恶化（`+0.00245/+0.03416/-0.02742`），1140M 三种子均改善（`-0.01892/-0.00644/-0.06136`）。所有运行保持 logits 逐位一致、身份隔离并未访问 locked test，但 1/2 confirmation 通过不足以支持可靠创新或 locked-test 资格。冻结 P3o-D/E/F，禁止对 1080M、1130M、1140M 重调或重跑。下一机制必须以跨身份 worst-combination 风险收缩为首要可证伪目标，而非平均参数 NMAE，并使用新的身份 seeds。详见 `docs/p3oe_invariant_residual_confirmation_results_20260802.md`。

## 2026-08-02 执行更新：P3o-G/H/I 有限有效组合风险验证

P3o-G 首次实现的组合 CVaR 在 inactive NaN 先进入 Smooth-L1 后再乘 mask，导致 `NaN * 0` 仍为 NaN；1230M/1240M 已冻结为实现无效，未作科学比较。P3o-H 使用有限有效布尔索引修复该缺陷，但独立进程收据不完整，1270M/1280M 冻结且不再续跑。

P3o-I 在全新 1330M/1340M identity 上，以每 seed 独立进程、成功/异常收据与 CUDA 释放完成相同的有限有效科学比较。三种子 2081/2082/2083 全部完成，身份交叠为零，分类 base state/logits 逐位一致，F 安全与检索门均通过，且没有生成、打开或使用 test/locked test。I1 equal-weight compound-CVaR 相对 F-base 的最坏组合 predicted-gate NMAE 在三 seed 分别为 `0.9669/0.9677/0.9590`，但相对更简单的 I0 shared-mean residual 为 `1.0150/1.0161/1.0146`，三 seed 同方向更差；整体 oracle/predicted-gate 相对 F-base 的 2% 改善门也未稳定通过。因此 P3o-I 冻结失败，不生成 1350M/1360M confirmation，禁止在 1330M/1340M 调整 CVaR 比例、权重、阈值或 checkpoint。详见 `docs/p3oi_durable_finite_combo_cvar_results_20260802.md`。

下一轮 P3o-J 已预注册为一个不同的信息机制而非 CVaR 调参：在同一冻结 shared delay residual 上，用 detached six-label posterior 显式选择有界 6x12 分量修正，并以 shared-mean residual 为强对照。新身份为 1450M/1460M（1470M/1480M 仅作预留确认），固定 seeds 为 2091/2092/2093；只有三 seed 同时优于 J0 的 oracle、predicted-gate 与 worst-combination NMAE，且不恶化 miss penalty 与 F 状态不变性，才允许生成确认数据。详见 `configs/p3oj_component_conditioned_residual_preregister_20260802.json`。

## 2026-08-03 执行更新：P3o-L 与 P3o-M 参数路线冻结

P3o-L 在全新 D 盘身份 `1530000000/1540000000` 上完成了 posterior-selected 6x12 有界残差与 shared-mean 强对照。它在整体 oracle/predicted-gate NMAE 上只有约 `0.16%--0.18%` 的三种子同方向改善、worst combination 约 `0.30%--0.35%` 改善，低于预注册 2% 门；且一个分类 base seed 未通过独立安全/检索门。因此不生成 `1550M/1560M` 确认身份，不把 component table 宽度、权重或选择规则在该数据上调优。详见 `docs/p3ol_component_conditioned_residual_results_20260803.md`。

P3o-M 以全新 `1570000000/1580000000` 身份重新检验 MFT count 的物理读出：候选将参考反卷积延迟响应压缩为严格局部 contrast 峰 token，控制使用相同 token 数、同参数 ordinal 网络的固定延迟分段读出。30 个 train-only 视图均以独立 MATLAB/HDF5 事务原子提交；两角色各 2,079 identities、检查 373 个历史文件后最大交叠为 0。C1 相对 C0 的整体和复合 MAE 在三种子均降低 `13.58%--16.94%`，Pearson 均提高约 `0.11`，说明稀疏事件选择有真实增量信息；但 low_count Pearson 为 `0.3057/0.2973/0.2967`，两 seed 未达预注册 `0.30`，故 `development_passed=false`。不生成 `1590M/1610M` 确认身份，不接入 predicted-gate 系统，也不在 1570M/1580M 调峰规则、token 数或损失。下一机制只能在新身份检验分辨率感知的区间预测/拒识与 coverage-risk，不能把 MFT count 表述为已可靠提取。详见 `docs/p3om_sparse_peak_ordinal_results_20260803.md`。

## 2026-08-03 执行更新：P3o-N/P3o-O 连续失败并冻结

P3o-N 在全新 `1720000000/1730000000` 身份上检验稀疏峰强度与最小间距是否可作为固定覆盖率下的分辨率感知拒识分数。C0/C1 共用完全相同的点预测与 selection-only 80% 阈值来源，但 C1 的总体与复合 MAE 在三种子均恶化约 8.5%，Pearson 也下降约 0.05；虽然 ultra-close 接受覆盖有方向性下降，已接受样本 MAE 反而变差。因此不得调整拒识公式、间距尺度、覆盖率或使用 `1740M/1750M` 确认身份。详见 `docs/p3on_resolution_abstention_results_20260803.md`。

P3o-O 在全新 `1760000000/1770000000` 身份上把下一有效稀疏峰的幅度差、对比度、尖锐度差和峰间延迟作为训练期关系 token，并以相同 146,983 参数、关系通道为零的 C0 控制作比较。身份交叉重叠为 0、对 393 个历史文件的最大重叠为 0；但 C1 相对 C0 的总体/复合 MAE 在三种子分别恶化约 `0.75%--1.12%`，相关性无稳定正增益，且一个 low-count Pearson 与一个 ultra-close MAE 门槛失败。不得调整关系 token 或使用 `1780M/1790M` 确认身份。P3o-M/N/O 共同说明：单峰稀疏选择本身有证据，但简单的相邻关系编码和后验拒识均不能成为可靠 MFT count 创新。详见 `docs/p3oo_adjacent_peak_relation_results_20260803.md`。

## 2026-08-03 执行更新：P3o-P 复相干信息近门槛失败并冻结

P3o-P 在新 `1800000000/1810000000` 身份上检验 MFT 的复 LFM 模板相干性是否为先前幅度峰 token 缺失的独立信息。C0/C1 保持完全相同的 16 个稀疏峰、146,983 参数网络、损失和训练；C1 仅新增反卷积 in-phase、quadrature、五样点 phasor coherence 和 in-phase coherence。两角色各 2,079 identities、对 402 个历史文件最大交叠为 0。C1 的总体/复合 MAE 在三种子均改善 `2.69%--4.35%`，ultra-close MAE 也同向改善，但 Pearson 仅 `+0.0090--+0.0180`，两个 seed 的 low-count Pearson 未达 0.30，且有 regime 在三 seed 中均恶化超过 2%，故仍未通过 5%/0.03 的冻结门。不得对相干描述符或 1800M/1810M 调参，不生成 `1820M/1830M` 确认身份。该结果支持“相干性可能含有弱增量信息”，但不支持将其表述为可靠参数创新。详见 `docs/p3op_complex_coherence_results_20260803.md`。

## 2026-08-03 执行更新：P3o-Q 全局非负支持曲线失败并停止 MFT count 专家路线

P3o-Q 在全新 `1840000000/1850000000` identity 上将 MFT 的正实幅度复 LFM 物理约束从局部峰扩展到完整 751 样点反卷积延迟窗。C0/C1 保持相同 16 个稀疏幅度 token、146,983 参数网络和训练；C1 仅添加候选计数 1--8 的全局非负 in-phase 支持/剩余复能量曲线。身份交叉重叠为 0、对 413 个历史文件最大重叠为 0。C1 的 low-count Pearson 三种子均为 `0.3392/0.3590/0.3657`，ultra-close MAE 也均改善，说明该全局物理约束不是无信息；但总体/复合 MAE 改善仅 `1.29%--3.31%`、Pearson 增量仅 `+0.0098--+0.0185`，未达到预注册 5%/+0.03 门槛。不得使用 `1860M/1870M`，也不得继续在 MFT count 的峰、相干或支持曲线上调参。至此冻结 MFT count 专家路线：未分离复合 I/Q 中，这些确定性物理描述符仅提供边缘增益，不能支持可靠参数创新或系统接回。详见 `docs/p3oq_nonnegative_support_curve_results_20260803.md`。
