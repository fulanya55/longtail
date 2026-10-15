# Route A H0l→H1m：三原子 OMP MFT 证据链

## 最终决定

关闭当前单脉冲协议下的**固定 LFM 匹配滤波 MFT 统计量家族**，包括全局有效支持度（H0j/H1k）与固定三原子复数 OMP 残余（H0l/H1m）。两者都能回答单 MFT 的副本数干预问题，却不能在弱 MFT 六类复合检测中提供正向、可归因的增量；不可再在这些已使用身份上搜索 ROI、原子数、熵阶、字典、正则化、归一化、分类头、阈值或融合。

## H0l：新配对物理干预通过

H0l 使用 640 对全新单 MFT 记录，3 与 8 副本共享 receiver-noise seed、JNR、公共到达延迟、8 个延迟候选与 8 个幅度候选，仅改变加入副本数。身份范围 `1058677744--1058678383` 与全部历史 train 来源零交集。

候选固定为正确已知 LFM 的三原子复数 OMP 残余能量比例；控制为相位置乱 LFM 的同字典、同原子数实现。二者均为 0 可训练参数、0 阈值、0 校准、0 融合。

| H0l 端点 | 值 |
| --- | ---: |
| 候选 AUROC（8 vs 3） | `1.00000` |
| 控制 AUROC | `0.54156` |
| 候选−控制 | `+0.45844` |
| 配对 bootstrap 95% CI | `[+0.42975,+0.48740]` |
| 配对 residual(8) > residual(3) | `1.00000` |

这说明在无共现成分时，三原子模型能准确保留“额外副本未被三原子解释”的物理剩余。

## H1m：新六类复合验证失败

H1m 使用独立 5,040 样本六类 train-only 来源，SHA-256 `4d1551141ea146b898ed8b973a91dd7625321826a1a5a231855e41f0a02843ab`，identity 为 `1068677743--1068682782`，fit/validation/evaluation=3024/1008/1008，和所有历史 train 身份零交集。评估只含 117 个弱 MFT 正例与 496 个其他五标签完全匹配的 MFT 缺失负例（31 个上下文）。

| H1m 弱 MFT 端点 | 候选 | 控制 | 候选−控制 |
| --- | ---: | ---: | ---: |
| AUROC | 0.34321 | 0.45535 | **−0.11215** |
| AUPR | 0.14001 | 0.17373 | **−0.03372** |
| AUROC 差 bootstrap 95% CI | — | — | **[−0.16967,−0.05471]** |
| AUPR 差 bootstrap 95% CI | — | — | **[−0.06730,−0.01627]** |

只有 1/5 receiver 子折 AUROC 差为正，其余四折均为负。所有身份、角色、零拟合和 test-v3 sealed 合同通过；失败来自检测机制本身，而非数据泄漏、阈值或随机训练不稳定。

## 解释

H0l 中所有匹配滤波原子都对应 MFT 副本；H1m 的复合样本中，LFM 相关结构、共同到达的强成分、噪声和其它干扰会同样占据该延迟字典。弱 MFT 时，OMP 选中的三原子不能被解释为“只属于 MFT”的原子，剩余能量也不再表示“额外 MFT 副本”。正确字典相对置乱字典反而更容易对复合背景做有结构的解释，因而造成负向排名。

因此下一个 MFT 方向不能继续增加匹配滤波的复杂度；需要改变信息条件或任务定义，例如多脉冲/跨脉冲一致性观测，或可验证地隔离共现成分的生成干预。二者都需要先做新的物理 H0，并且必须使用新的身份来源。

## 边界

test-v3 状态始终为 `access_authorized=false`、`analysis_allowed=false`、`used=false`。H0l/H1m 不影响 RawSaw SawFM 或冻结 ISRJ aligned-periodicity 的窄证据链，也不构成六标签网络改进或 final-test 授权。

## 机器产物

- H0l：`outputs/tapc/route_a_h0l_mft_omp_residual/route_a_h0l_mft_omp_residual_summary.json`
- H0l 审计：`outputs/tapc/route_a_h0l_mft_omp_residual_generation/paired_identity_audit.json`
- H1m：`outputs/tapc/route_a_h1m_mft_omp/route_a_h1m_mft_omp_summary.json`
- H1m 审计：`outputs/tapc/route_a_h1m_mft_omp_composite_generation/identity_audit.json`
