# P1e：20–25 dB过渡区物理特征充分性与Oracle安全接管筛查

## 1. 结论

本轮冻结P1c/P1d主网、U-Net和MFT/VFT/ISRJ专家，只检验一个问题：在P1d风险集中的20–25 dB过渡区，目标特异物理特征是否能比现有概率特征更好地区分“专家有收益”与“专家会制造假阳性”。

正式三种子结果为：**MFT、VFT、ISRJ均未通过预注册的3/3 seeds晋级门槛，不增加路由器容量，不进入P2。**

主要发现：

1. MFT、VFT在20–25 dB的oracle安全接管Recall上界只有2.40%和6.04%，即使存在完美路由，也无法达到10%的系统增益目标；主要矛盾仍是专家能力不足，而不是路由器没有找到正确样本。
2. ISRJ存在足够的oracle空间，平均Recall上界为15.42%；物理特征融合将实际路由Recall增益从4.27%提高到8.85%，但最大FPR恶化达到2.26%，未满足0.5%的安全限制。
3. 现有概率特征对假阳性风险的AUROC已经达到0.961–0.993。加入物理特征后的平均AUROC增量只有MFT +0.0018、VFT +0.0009、ISRJ +0.0171，均未达到预注册的+0.03。这说明当前瓶颈不是简单缺少更多路由输入维度。

## 2. 证据边界与数据协议

- 只使用P1d新身份数据的`train`文件；未发现或读取其已消费的independent test。
- 使用经审计的float32样本分块缓存，模型输入数值与原始MAT读取后的float32值一致。
- 只使用20 dB和25 dB两个风险过渡档。
- 六类顺序固定为`[FMB, SawFM, SinFM, MFT, VFT, ISRJ]`。
- 局部三输出固定解释为`[MFT, VFT, ISRJ]`。
- 63种非空标签组合每种20个身份，逐组合固定划分为：
  - fit：10个，共630个身份；
  - calibration：5个，共315个身份；
  - probe：5个，共315个身份。
- 同一身份的20/25 dB视图始终进入同一角色。
- 三种子使用相同身份角色，仅模型seed不同。
- `independent_test_used=false`、`confirm_used=false`、`locked_test_used=false`。

## 3. 冻结模型与特征对照

### 3.1 冻结模块

- 六类主网：MLCNN分类 + IQ-Grouped参数任务路由模型的parameter checkpoint；
- 分离器：已有复数STFT U-Net；
- MFT专家：U-Net局部输出0；
- VFT专家：`stack_trajectory_leaf15`重建规则；
- ISRJ专家：`stack_all_periodicity`重建规则；
- 不重新训练任何神经网络或物理专家。

### 3.2 三组路由输入

| 变体 | 输入 |
|---|---|
| probability-only | 六类主网概率、三专家概率、目标分歧、阈值距离、其他类别置信度摘要 |
| physical-only | MFT延迟/自相关特征、VFT瞬时频率轨迹、ISRJ周期/包络自相关特征 |
| probability+physical | 上述两类特征拼接 |

风险与收益模型均固定使用HistGradientBoosting：150 iterations、learning rate 0.05、7 leaves、L2=1、balanced class weight。

## 4. Oracle目标与实际路由

逐样本定义：

- benefit：专家BCE相对主网至少降低0.05 nat；
- false-positive risk：真实标签为负，专家概率高于主网且越过原主网阈值；
- oracle accept：有benefit且不存在false-positive risk。

实际路由采用：

1. calibration上选择风险否决阈值，目标风险召回不低于0.90；
2. 固定`alpha=0.75`；
3. 在预注册benefit cutoff网格中，选择FPR增量不超过0.005、AUPR增量不低于-0.005时Recall增益最大的候选；
4. 在identity-disjoint probe上一次评价。

## 5. 三种子汇总结果

| 目标 | Oracle Recall上界 | 概率风险AUROC | 融合风险AUROC | AUROC增量 | 概率路由Recall增益 | 融合路由Recall增益 | 融合最大FPR增量 | 晋级 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MFT | +0.02396 | 0.99304 | 0.99482 | +0.00179 | +0.01146 | +0.00937 | +0.01290 | 否 |
| VFT | +0.06042 | 0.97745 | 0.97835 | +0.00090 | +0.01042 | +0.01042 | +0.00645 | 否 |
| ISRJ | +0.15417 | 0.96054 | 0.97760 | +0.01707 | +0.04271 | +0.08854 | +0.02258 | 否 |

所有目标的`seed_pass_count=0/3`，`any_target_promoted=false`。

## 6. 分种子诊断

### 6.1 MFT

- Oracle Recall增益：2.19%、2.81%、2.19%；
- 融合风险AUROC：0.9953、0.9933、0.9959；
- 融合实际Recall增益：0.94%、1.25%、0.62%；
- seed 2042/2043 FPR分别恶化0.97%/1.29%。

判断：风险排序已经非常好，但专家能够安全修复的阳性样本本身太少。继续增加路由容量不会弥补专家上界不足。

### 6.2 VFT

- Oracle Recall增益：4.69%、5.94%、7.50%；
- 融合风险AUROC：0.9742、0.9825、0.9783；
- 融合实际Recall增益：1.25%、0.31%、1.56%；
- 物理特征对风险AUROC的增量有一个seed为负。

判断：VFT专家在20–25 dB具有少量收益，但距10%目标仍有明显差距；瞬时频率特征主要已经反映在专家概率中，重复加入路由输入没有产生新的可分信息。

### 6.3 ISRJ

- Oracle Recall增益：18.13%、16.88%、11.25%；
- 融合风险AUROC：0.9761、0.9789、0.9778；
- 融合实际Recall增益：11.88%、10.00%、4.69%；
- FPR变化：+0.65%、+2.26%、-2.26%。

20 dB是主要收益来源：seed 2041/2042分别提升16.88%和15.00%；25 dB增益普遍较小。seed 2042在两个功率档均产生FPR恶化，导致整体安全门失败。

判断：ISRJ专家具备值得保留的oracle收益，但当前问题是风险保证和阈值迁移，不是继续添加物理特征或更复杂分类器。

## 7. 晋级门槛检查

单seed必须同时满足：

1. 融合风险AUROC不低于0.70；
2. 融合benefit AUROC不低于0.65；
3. 相对概率特征风险AUROC提高至少0.03；
4. probe风险召回不低于0.80；
5. 至少保留50%的真实benefit样本；
6. 实际Recall增益相对概率路由至少增加0.02；
7. FPR增量不超过0.005。

虽然绝对风险/收益AUROC普遍很高，但三目标都没有达到物理特征增量门槛；MFT/VFT还受到oracle上界不足限制，ISRJ受到FPR不稳定限制。因此没有目标满足3/3 seeds晋级。

## 8. 失败原因与路线决策

### MFT/VFT

停止“在现有专家上继续改路由”的路线。下一步如果继续，必须先提高专家本身的oracle上界：

- MFT：参考匹配滤波、峰归属、延迟峰集合与ordinal数量头；
- VFT：时延对齐、dechirp、残余相位斜率或更长相干观测。

只有新专家在20–25 dB oracle-positive筛查中实现至少10% Recall/MAE收益，才重新讨论路由。

### ISRJ

保留为唯一具有足够transition-region oracle收益的候选，但本轮不晋级。若在全新development-v2继续，应只研究：

- 更大的身份级风险校准样本；
- 目标特异的Clopper-Pearson/容忍区间缓冲；
- 优势置信下界大于0时才接管；
- 最坏功率或最坏组合风险约束。

不建议继续增加HGB深度、额外物理特征或端到端训练。

## 9. 产物与验证

- 实现：`cj0724/p1e_transition_physical_sufficiency.py`
- 扩展物理特征输出：`cj0724/integrate_swept_experts_with_six_class.py`
- 测试：`tests/test_p1e_transition_physical_sufficiency.py`
- 正式输出：`outputs/p1e_transition_physical_sufficiency_20260731/`
- 冻结协议：`outputs/p1e_transition_physical_sufficiency_20260731/protocol.json`
- 汇总：`outputs/p1e_transition_physical_sufficiency_20260731/summary.json`
- 完整结果：`outputs/p1e_transition_physical_sufficiency_20260731/results.json`
- 逐样本结果：`seed_2041|2042|2043/probe_per_sample.npz`
- 正式运行耗时：187.36秒；
- PY10全量回归：89 passed；
- summary SHA256：`5B7FA38163C387D60F3F688F75F5ECE44373DC3D53318E98EFF13CCB0BC8BA9A`；
- results SHA256：`9FC663D437DF50E69E1B184F0AE641AD0220DC49A2B5B0E7118C138F451F2D23`。

