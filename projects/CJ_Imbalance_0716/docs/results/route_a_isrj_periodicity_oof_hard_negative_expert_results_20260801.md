# Route A ISRJ 周期专家 OOF 难负重训结果

## 判定

组件门禁通过 6/9，整体失败，不得重新接回 H1 融合头。

aligned-hard 相对 aligned-base 的 early matched AUROC 仅增加 0.001097，95% bootstrap CI 为 `[-0.004960,0.006242]`，未达到 0.010 门槛；五个种子只有 3 个为正。AUPR 平均增加 0.002097，但跨种子不稳定。与此同时，aligned-hard 相对完全同协议的 shuffled-hard AUROC 高 0.079812，95% CI 为 `[0.062079,0.096586]`，五个种子全部为正。

因此，本轮不是否定周期机制，而是表明在当前 310 维周期表示与 7 叶 HGB 容量下，OOF 难负加权几乎没有额外可提取收益。

## 逐种子结果

| seed | H0 aligned AUROC | H1 hard AUROC | H1-H0 | H0 AUPR | H1 AUPR | H1-H0 | shuffled-hard AUROC | H1-S1 | val-early gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2039 | 0.6150 | 0.6181 | +0.0031 | 0.2443 | 0.2330 | -0.0113 | 0.5327 | +0.0853 | 0.0854 |
| 2040 | 0.6011 | 0.5924 | -0.0087 | 0.2247 | 0.2193 | -0.0054 | 0.5375 | +0.0549 | 0.2126 |
| 2041 | 0.6285 | 0.6382 | +0.0098 | 0.2692 | 0.2970 | +0.0277 | 0.5342 | +0.1041 | 0.1692 |
| 2042 | 0.6624 | 0.6595 | -0.0029 | 0.2716 | 0.2677 | -0.0039 | 0.5624 | +0.0971 | -0.0279 |
| 2043 | 0.6065 | 0.6108 | +0.0043 | 0.2509 | 0.2542 | +0.0034 | 0.5531 | +0.0576 | 0.0994 |

每个种子的 aligned fit matched view 含 258—270 个负样本，OOF 最高四分位产生 65—68 个难负，选择过程有效且无训练内分数泄漏。失败不能归因于难负集合为空或协议错误。

## 表示层根因

当前 310 维特征主要由 complex/envelope ACF 的幅度池化、频谱能量分箱和少量周期峰统计组成。它能检测“存在周期结构”，因此稳定优于 block shuffle；但它大量丢弃了复数相关的相位信息和跨周期复制一致性，难以区分：

- ISRJ 的截取—重发副本；
- FMB/VFT 产生的规则调制周期；
- 多分量叠加造成的包络伪峰。

难负加权只能重新划分已有特征空间，无法创造“延迟副本是否保持局部复波形一致”的证据，所以增益饱和。下一结构改进应新增低容量 copy-consistency 特征，而不是扩宽 HGB 或继续调样本权重：在候选重复时延处计算分窗复相关幅度稳定性、相位圆一致性、谐波时延一致性以及复相关/包络相关比值，并保留 block-shuffle 机制控制。

## 证据边界与产物

- 数据：development-v3，种子 2039—2043。
- 5 折 OOF 只用于 fit-role 难负挖掘；validation/early 未参与训练。
- test-v3 未访问、未分析、未使用。
- 预注册：`docs/plans/route_a_isrj_periodicity_oof_hard_negative_expert_plan_20260801.md`。
- 实现：`experiments/run_route_a_isrj_periodicity_oof_hard_negative_expert.py`。
- 测试：`tests/test_route_a_isrj_periodicity_oof_hard_negative_expert.py`，1 passed。
- 输出：`outputs/tapc/route_a_isrj_periodicity_oof_hard_negative_expert`。
- 汇总 SHA256：`76AC16D5DF9AEC5F8096FE27153FF2BB90024A5105C6919F3ADAA70113F1D4BA`。

