# Route A IP-CVE：P0 后扩展观测协议准入审计

日期：2026-08-07  
状态：`completed_no_run`  
决策：`no_run_close_local_ipcve_mechanism_external_observation_required`

## 1. 结论

**当前项目内没有一个满足非重叠准入门、可以直接继续生成数据和训练模型的本地扩展观测方案。**

这不是算力、训练时长或网络容量不足，而是三条可执行本地候选均已被既有证据覆盖：

- MFT：长延迟孔径的物理量可见，但相同信息的学习接回失败；
- VFT：交错 PRI 的真实时间戳物理量可见，但学习模型未战胜同容量时间戳置乱控制；
- ISRJ：单通道周期/相关统计在复合背景中不能稳定完成弱成分的标签归因。

因此，本轮按预注册规则执行 `no_run`：没有生成新身份数据，没有训练新模型，也没有消费 confirmation、locked、test-v2 或 test-v3。继续增加 adapter、attention、MLP、损失权重或阈值只会构成对已关闭机制的换名重试。

## 2. MFT 证据链

| 证据 | 结果 | 含义 |
|---|---:|---|
| H0u 正确 LFM 三延迟带相对相位置乱控制的 AUROC 增益 | `+0.407805` | 1 ms、三组分离副本下物理可观测性成立 |
| H1u 学习接回 | `passed=false` | 固定三峰残差没有转化为稳定弱 MFT 分类增量 |
| H2v 条件互补准入 | `admitted=false` | 一种子 B0 inversion rate 仅 `4.07% < 5%`，候选未训练 |

H1u 中 M1 物理残差自身弱正/缺失 AUROC 很高，但 B0 在四个种子上已接近饱和；残差改变共享多标签优化轨迹并带来非目标退化。由此，重新调整 delay bands、residual cap、backbone、loss 或 gate 都属于已关闭 H0u/H1u/H2v 机制族。

## 3. VFT 证据链

| 证据 | 结果 | 含义 |
|---|---:|---|
| H1b 五背景真实时间戳相对置乱控制 AUROC 增益 | `+0.115471` | 交错 PRI 在物理层提供 VFT 可归因信息 |
| H2 C3 相对 C0 弱 VFT AUROC | `+0.014150` | 有小信号，但低于 `+0.030` 门 |
| H2 C3 相对同容量 C4 弱 VFT AUROC | `-0.000405` | 真实时间顺序没有战胜置乱容量控制 |
| H2 C3 相对 C0 macro-AUPR | `-0.001862` | 共享六标签性能退化 |

所以，真实时间戳统计“有物理信息”与“能形成网络创新增量”是两个不同命题。H2 已否定当前加性专家接回；继续搜索时间戳置乱方式、ROI、专家宽度、损失权重、训练轮数或阈值不具备新机制资格。

## 4. ISRJ 证据链

| 协议 | true AUROC | 相对破坏控制 | paired weak > absent |
|---|---:|---:|---:|
| P1 | `0.577544` | `+0.082222` | `0.440000` |
| P2 | `0.555100` | `+0.056144` | `0.513333` |

P1/P2 的时间顺序统计都明显高于破坏控制，但 absent 本身也具有很高周期/相关响应。真正失败的是弱 ISRJ 是否能归因到 ISRJ 标签：

| nuisance 标签数 | P1 weak-absent | P2 weak-absent |
|---:|---:|---:|
| 1 | `+0.0913` | `+0.1038` |
| 2 | `-0.0005` | `+0.0093` |
| 3 | `-0.0068` | `+0.0004` |
| 4 | `+0.0000` | `-0.0034` |
| 5 | `+0.0029` | `-0.0019` |

当 matched absent 中有两个及以上 nuisance 标签时，弱增量基本归零；强 ISRJ 仍为正。这是单通道复合混合的可见性/归因边界，而不是换分类头可以修复的读出容量问题。

## 5. 当前代码能力审计

只读检查了 P1、P2、H0u、H0q、H1a、H1b 六个相关 active generator。它们均生成一条复 I/Q 观测及标签/参数/seed 元数据；在这组六个直接相关 generator 中，没有声明：

- 第二同步 receiver；
- 非 oracle 的校准参考通道；
- 受控 receiver setting 的第二观测；
- `iq_aux` 或 `iq_reference` 推理输入。

该结论严格限定于本次审计的六个活动生成器，不声称整个 `E:\LJQ_code` 从未存在其他多通道代码。就当前 Goal 的可复现实验入口而言，新增第二物理观测需要先明确新的传感协议和生成/采集契约，不能从现有 absent 反事实直接伪造成推理参考通道。

## 6. 准入矩阵

| 候选 | 是否与历史重叠 | 当前可直接执行 | 决策 |
|---|---|---|---|
| MFT 长窗/LFM 残差 | 是：H0u/H1u/H2v | 否 | closed |
| VFT 交错 PRI/时间戳专家 | 是：H0q/H1a/H1b/H2 | 否 | closed |
| ISRJ 单通道周期/相关专家 | 是：H0v/P1/P2/既有周期线 | 否 | closed |
| 额外校准通道或受控 receiver setting | 机制上非重叠 | 当前协议不存在 | external state required |

`new_local_h0_candidate_admitted=false`。

## 7. 下一项唯一允许工作

如需重新启动统一三目标创新，必须先独立定义并获得一个**非 oracle 的新增物理观测协议**，例如：

1. 同步主通道与校准辅助通道，辅助通道由真实传感几何/接收链路获得，而不是直接读取仿真的 matched absent；
2. 或同一目标在两个预先固定、可部署的 receiver settings 下获得观测，并冻结两设置的传递关系；
3. 或真实/半实物/HIL 数据提供可复核的弱 ISRJ 标签归因信息。

新协议必须先重启 Stage -1，再在全新 identities 上执行三目标零拟合 H0；必须包含 channel swap、transfer perturbation、time permutation 和 matched nuisance 控制，并先证明 MFT/VFT/ISRJ 三者的弱正相对缺失增量都可归因。H0 未通过前禁止训练网络。

在该外部状态出现前，论文主张应收缩为：

- 当前单视图合成协议下统一 IP-CVE 被可靠否定；
- MFT/VFT 的长窗物理可观测性与网络增量是不同问题；
- 现有可靠正证据仅能按标签局部组件分别陈述，不得包装成统一双轴长尾方法已经成立。

## 8. 工件与封存状态

- 机器审计：`outputs/route_a_ipcve/post_p0_observation_admission/admission_audit.json`
- 审计 SHA256：`4693e357ce86e219033aec08507019d3dd79626fdbcba00f32a1de1fe0c7ae8e`
- 运行收据：`outputs/route_a_ipcve/post_p0_observation_admission/run_receipt.json`
- 测试：`8 passed`
- `locked_test_used=false`
- `test_v2_used=false`
- `test_v3_used=false`

