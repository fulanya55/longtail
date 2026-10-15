# Route A：MFT/VFT 新架构准入审计与窄 ISRJ 证据转向

日期：2026-08-03  
性质：机制重叠审计与证据包规划；**没有生成数据、训练模型、重新校准、重新选阈值或读取任何测试 I/Q。**

## 结论先行

本轮提出但**不予执行**的 MFT「盲时频脊线数量/持续性」候选，不是当前项目中可接受的新 H0。

1. 若用已知发射 LFM 的 chirp-rate 将时频脊线去斜，多个时延副本会变为多个频率偏移；该表示与已关闭的 MFT 匹配滤波 delay-profile/RAAT delay-topology 属于同一个时延可观察量的坐标变换，不能借图像化或峰数重命名为新机制。
2. 若不使用 chirp-rate，只统计 STFT 的峰数、连通性或形态，则它不是 MFT 特异的物理证据，而是普通时频容量/形态分支；该类扩容已被 `route_a_valid_target_heterogeneity_restart_gate_20260802.md` 排除。
3. 若将多副本线索改写为倒谱、延迟谱或非零延迟相关，它仍属于 H0p 已失败的全局 delayed-copy correlation complexity 家族；H0p 的 candidate-minus-control 只有 `+0.03316`，低于预注册 `+0.10` 门。

因此，现阶段没有一个同时满足“机制距离、可部署性、信息破坏对照、未消耗身份”的 MFT 或 VFT H0。为避免在近邻统计变体上再次消耗新身份，MFT/VFT 不启动新数据生成或网络训练。

可继续积累的可靠主张应收缩为：**冻结 Focal-36 + H1 线性分类头 + RawSaw 安全旁路上的 ISRJ aligned-periodicity 增量，在新的高副本参数区和 AR(1) 接收机噪声条件下均保持弱 ISRJ 排序和召回增益。** 这不是完整六标签 VTSE-VBF 的完成证明，也不授权开启 `test-v3`。

## 1. 已关闭机制与本轮候选的等价性

| 被考虑的读出 | 若如何实现 | 与既有机制的关系 | 决定 |
| --- | --- | --- | --- |
| LFM 去斜后的多频率峰/脊线数量 | 固定 chirp-rate、局部 STFT、候选为峰数或持久峰数，错误 chirp 为控制 | 延时副本在去斜后对应频移；与正确参考匹配延迟图同属 delay-coordinate 变换。RAAT-MFT 已因该重叠于训练前撤销。 | 不准入 |
| 盲 STFT 峰数、时频脊线连通性 | 不使用参考、以通用局部峰/形态为输入 | 不是 MFT 特异；等价于已禁止的普通 STFT 分辨率/形态容量扩展，且不能建立目标专属因果控制。 | 不准入 |
| 倒谱/延迟谱/相关支持度 | 以多副本延迟造成的频谱纹波或相关峰为依据 | 仍是 delayed-copy 自相关复杂度的近邻。H0p 已显示相位索引置换控制同样高，不能归因于 MFT。 | 不准入 |
| VFT 的去斜后脊线方向或残余相位 | 正确 LFM 去斜后再读相位/方向 | H0i/H1j 的同一参考去调频残余机制；H1j 的 identity bootstrap CI 跨零，已关闭。 | 不准入 |

## 2. 既有失败所给出的科学限制

- H1k/H1m：固定单脉冲参考对齐的 MFT support/dictionary 在新六标签复合背景中反向，不能把局部有效的单 MFT 可辨识性外推为复合归因。
- H0o：跨 PRI 相干相对幅度控制有正差异，但绝对 AUROC `0.58066 < 0.60`；不能以更换 top-k、PRI、ROI 或融合方式继续搜索。
- H0p：单 PRI、无已知参考的相关复杂度对 3-vs-8 副本敏感，但其信息破坏控制同样敏感，排除了“纯时间相关复杂度就是 MFT 证据”的解释。
- H1j：VFT 正确去斜固定分数的 AUROC 和 AUPR identity bootstrap CI 均跨零；不能改窗口、归一化、频率范围或 readout 后重启。

它们共同说明的不是“主干还不够宽”，而是当前合成复合协议下可见的 MFT/VFT 局部统计会被背景成分解释，尚未建立对目标标签的可归因性。

## 3. 文献冲突检查的边界

外部全文检索网关在本轮返回 HTTP 404；未把标题、摘要或元数据当作网络/损失/性能事实。通过 Crossref 仅核对了下列元数据，均标记为 `[T]`，只说明相关主题存在，不能支撑本项目的新颖性主张：

- `10.1088/1742-6596/1871/1/012079`：false-target jamming 的时频特征提取主题 `[T]`；
- `10.1109/tsp.2013.6614028`：LFM chirp 的局部 chirp-rate 时频估计主题 `[T]`；
- `10.3390/s21165415`：LFM 检测/时频率分布估计主题 `[T]`。

这反而加强了保守决定：即使没有项目内重叠，通用时频脊线检测也不足以构成可独立声称的创新。未来如取得可逐节核验全文，需补齐输入、目标、控制、训练和局限后，才可重启任何新机制筛选。

## 4. 当前可推进的窄证据链

| 证据角色 | B2 相对冻结 B0 的 ISRJ 结果 | 允许结论 | 不允许结论 |
| --- | --- | --- | --- |
| development-v3 | weak AUROC `+0.070336`、weak recall `+0.092397` | 开发中出现 aligned-periodicity 增量 | 不是外部泛化 |
| H0g，高副本参数外推 | weak AUROC `+0.099646`、weak recall `+0.071252`，均 5/5 正 | 新身份/高副本条件下的冻结复现 | 不修复 MFT/VFT，不是 final test |
| H0h，AR(1) 噪声 | weak AUROC `+0.080371`、weak recall `+0.102703`，均 5/5 正 | 接收机相关噪声下的冻结复现 | 不是所有噪声模型鲁棒性 |
| H0g/H0h 共标签只读审计 | 两源全部 9 个 MFT/VFT 分层 weak-AUROC 下 CI 均大于 0 | 排名增益不限于单一 MFT/VFT 共标签 | 阈值召回/FPR 并非所有上下文均安全 |

所有四项都必须保留 `external_refit=false`、`locked_test_used=false`、`test_v3_used=false`，并随证据包记录输入 SHA-256。

## 5. 进入下一项新模型实验的硬门

任何未来 MFT/VFT 新架构必须在生成数据前书面满足以下全部条件：

1. 给出与 reference-delay、dechirp residual、global autocorrelation、STFT morphology 均不等价的数学可观测量；
2. 给出能破坏该可观测量但匹配维度、能量、容量和优化预算的控制；
3. 不读取参数真值、功率、seed、标签查询或已有 logits；
4. 在全新 receiver-noise identity 的单标签 H0 中先通过绝对可辨识性及 candidate-minus-control 门；
5. 再在另一新身份的复合匹配负样本 H1 中完成 five-seed、paired identity CI 和强正安全门；
6. 仅当 MFT、VFT、ISRJ 三者均有独立 H1 通过时，才允许设计六标签 H2；此前不得访问 `test-v3`。

## 6. 本轮产物

- 本准入审计：`docs/reports/route_a_mft_vft_architecture_admission_audit_20260803.md`；
- 只读 ISRJ 证据包构建器：`experiments/build_route_a_isrj_evidence_bundle.py`；
- 证据包机器可读清单：运行后写入 `outputs/tapc/route_a_isrj_evidence_bundle_20260803/evidence_bundle.json`。

