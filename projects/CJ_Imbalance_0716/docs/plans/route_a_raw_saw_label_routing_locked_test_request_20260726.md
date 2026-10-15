# 路线 A：RawSaw+H2Others locked-test 申请包（2026-07-26）

## 申请状态

**待用户明确批准；当前未执行 locked test。**

Preflight冻结审计已通过，状态为`ready_for_authorization`。五个seed的F2/H2/Raw checkpoint、H2 temperature、H2六标签threshold、Raw Platt系数/截距和Raw统一threshold均已固化，不需要在最终测试时重新拟合任何参数。

## 候选方法

`RawSaw+H2Others`：

- FMB、SinFM、MFT、VFT、ISRJ：冻结 H2 概率与各自 threshold；
- SawFM：冻结 Raw frontend，ordinary calibration 上拟合 Platt，并使用 ordinary threshold + 40/45/50 dB power threshold 共同选择的统一阈值；
- 不使用融合网络、oracle power、组合标签或 test-time 阈值选择。

## Development 证据

- Raw 五种子确认：4/5 seeds 通过预设门禁；
- 标签级路由：5/5 seeds 通过；
- macro-AUPR：0.8667 → 0.9001；
- macro-F1：0.7573 → 0.8027；
- SawFM recall：0.4870 → 0.8813；
- exact match：0.2280 → 0.2747；
- 其他五类 FPR 变化：0；
- identity overlap：0；
- locked test used：false。

## 已知风险

- seed 2039 ordinary early FPR 为 0.2806，超过单 seed 0.20 门禁；五种子仍按预注册的 4/5 规则通过；
- 不允许根据 locked-test 结果修正该 seed 阈值；
- 2039/2040 与 2041–2043 的 Raw checkpoint 来自两个已审计抽样协议，网络和优化参数相同，但训练角色来源应在论文中披露；
- 当前没有批准使用 ensemble，因此 locked test 默认应按五个独立 seed 分别报告均值、标准差和逐 seed 结果。

## 批准后的一次性执行规则

1. 先记录 locked-test 数据文件哈希和当前汇总 SHA256；
2. 只加载本申请列出的五组冻结 F2/H2/Raw checkpoint；
3. calibration 和 threshold 不得接触 locked test；
4. 输出每 seed 的 macro/micro AUPR、macro-F1、Hamming loss、exact match、逐标签 recall/FPR/ECE、组合指标和 power 指标；
5. 同时输出 H2All 对照；
6. 无论结果好坏，运行一次后即关闭测试集，不重新训练或重新选阈值；
7. 保存 predictions、metrics、checkpoint hashes、data hash、environment、receipt 和完整命令。

## 冻结依据

- Development 汇总：`outputs/tapc/route_a_raw_platt_five_seed_label_routing/route_a_raw_platt_five_seed_label_routing_summary.json`
- 汇总 SHA256：`1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC`
- 结果报告：`docs/results/route_a_raw_platt_five_seed_label_routing_results_20260726.md`
- 冻结推理包：`outputs/tapc/route_a_raw_saw_locked_test_preflight/frozen_inference_bundle.json`
- 冻结推理包 SHA256：`F5FB713FE0C8EF351C6CDD42304E5D1541C955D27B28CBBFDB5B8B03B55B9489`
- Preflight审计：`outputs/tapc/route_a_raw_saw_locked_test_preflight/preflight_audit.json`
- Preflight审计 SHA256：`B53350B746E6BA83B6462521027699C03CA2F1692FD22F9AE0D75623535E754A`
