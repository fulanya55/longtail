# 路线 A：早期复数 SawFM 证据源筛选与固定确认结果（2026-07-25）

## 1. 总结论

本轮发现并确认了一个此前被 Stage2 以后表征显著削弱的 SawFM 证据源：

> RMS 归一化原始 I/Q 物理时序与 F2 第一复数残差层特征的组合 P6，在组合匹配 hard negative 上稳定保留强 SawFM 可辨识性；证据随主干加深到 Stage2 明显下降。

发现筛选与固定身份确认均为 `3/3` 种子通过。当前决策为：

`design_small_raw_stage1_sawfm_frontend_keep_h2_as_decision_baseline`

这允许下一阶段实现小容量 Raw+Stage1 SawFM 独立前端，但不允许直接替换 H2、不允许解冻 F2，也不代表已经完成论文级新种子确认。

## 2. 发现阶段

### 2.1 协议

- F2/H2/S2/C2/A0 全部冻结；
- fit、validation、development 身份隔离；
- 正样本：SawFM 存在且含扫频复合背景；
- 负样本：其他五类标签组合相同但 SawFM 缺失的 hard negative；
- 功率域：40、45、50 dB；
- 种子：2041、2042、2043；
- locked test：未使用。

### 2.2 聚合结果

| 候选 | 证据源 | validation mean AUROC | development mean AUROC | development worst AUROC | development mean AUPR |
|---|---|---:|---:|---:|---:|
| P0 | H2 logit | 0.4406 | 0.4262 | 0.3973 | 0.4432 |
| P1 | visibility 统计 | 0.8930 | 0.8818 | 0.8235 | 0.8629 |
| P2 | 原始 I/Q 物理时序 | 0.9502 | 0.9353 | 0.9081 | 0.9409 |
| P3 | Stage1 复数特征 | 0.8620 | 0.8646 | 0.7753 | 0.8434 |
| P4 | Stage2 复数特征 | 0.7766 | 0.7622 | 0.7287 | 0.7522 |
| P5 | Stage3 复数特征 | 0.8376 | 0.8284 | 0.7933 | 0.8096 |
| P6 | 原始 I/Q + Stage1 | **0.9551** | **0.9567** | **0.9353** | **0.9577** |

P6 相对 P1 的 development mean AUROC 提高 `+0.0749`。逐种子 P6 mean/worst/50 dB AUROC：

| 种子 | mean | worst | 50 dB |
|---:|---:|---:|---:|
| 2041 | 0.9532 | 0.9402 | 0.9402 |
| 2042 | 0.9636 | 0.9348 | 0.9348 |
| 2043 | 0.9534 | 0.9309 | 0.9309 |

预注册发现门为 `3/3` 通过。

## 3. 固定确认

发现阶段三个种子都选择了 P6 的 `C=0.1`，因此确认阶段冻结：

- 948 维 Raw+Stage1 描述；
- L2 Logistic Regression；
- `C=0.1`；
- fit role 不变；
- scale role 作为未参与 P6 选参的身份确认集。

同时使用固定 P1 `C=0.01` 和标签置乱 P6 对照。

| 方法 | scale mean AUROC | scale worst AUROC | scale mean AUPR | development复现 AUROC |
|---|---:|---:|---:|---:|
| C0 固定 P1 | 0.8959 | 0.8404 | 0.8840 | 0.8818 |
| C1 固定 P6 | **0.9576** | **0.9317** | **0.9539** | **0.9567** |
| C2 标签置乱 | 0.5140 | 0.4896 | 0.5236 | 0.5151 |

逐种子 C1 scale mean/worst/50 dB AUROC：

| 种子 | mean | worst | 50 dB | 相对 C0 mean 增量 |
|---:|---:|---:|---:|---:|
| 2041 | 0.9582 | 0.9340 | 0.9340 | +0.0714 |
| 2042 | 0.9454 | 0.9150 | 0.9150 | +0.0451 |
| 2043 | 0.9693 | 0.9461 | 0.9461 | +0.0686 |

固定确认门同样为 `3/3` 通过。

## 4. 对主干问题的新判断

结果形成了清晰的层级证据链：

1. 原始 I/Q 已有很强的 SawFM 弱证据，P2 development AUROC 为 0.9353；
2. 第一复数层仍保留较多信息，但单独使用低于原始 I/Q；
3. Raw+Stage1 互补后达到 0.9567；
4. 到 Stage2 降至 0.7622；
5. H2 输出进一步降至 0.4262。

因此当前 SawFM 弱分量问题不能再归因于“输入中没有信息”。更准确的根因是：

- 共享复数主干在 Stage1→Stage2 的降采样与残差变换中破坏或稀释了部分 SawFM 周期/幅相微结构；
- 后续共享六类优化没有恢复该证据；
- Stage2 专家、visibility 校准和 reliability 头只能在已经受损的表示上补救，因此存在性能上限；
- P6 的优势不是总功率捷径，因为输入经过 RMS 归一化、正负样本按其他标签组合匹配，且标签置乱结果接近随机。

## 5. 下一阶段网络约束

下一阶段应实现小容量 SawFM 独立前端，而不是扩宽整个主干：

```text
RMS-normalized I/Q
├─ 冻结 F2 主干 → H2 六类输出（继续作为正式基线）
└─ SawFM early frontend
   ├─ Raw-IQ physics branch
   │  ├─ magnitude / magnitude-difference
   │  ├─ normalized phase increments, lags 1/4/16
   │  └─ multi-scale temporal conv, kernels 7/31/63
   ├─ frozen Stage1 complex branch
   │  └─ low-rank 1x1 adapter + depthwise temporal conv
   ├─ statistics safety path: mean/std/max/top-k
   └─ normalized SawFM evidence logit
```

训练时先保持 F2 和 Stage1 完全冻结，只训练早期前端。监督继续使用同一身份多功率正样本、组合匹配 hard negative，并加入 root/40/45/50 的域平衡。只有独立前端在新种子确认通过后，才研究最大幅度 `0.20*tanh(z)` 的有界残差接回；接回前 H2 仍是正式决策基线。

## 6. 证据边界

已经验证：

- P6 在三个训练种子的独立 development 和未参与选参的 scale 身份上稳定有效；
- 50 dB AUROC 均高于 0.91；
- 标签置乱对照接近随机；
- locked test 未使用，身份重叠为 0。

尚未验证：

- 新模型初始化种子或新 F2 checkpoint 的完全独立确认；
- 可训练小前端能否复现固定描述符的 oracle 表现；
- 接回六类分类器后能否改善 SawFM，同时保持 macro-AUPR、FPR 和其他五类非劣；
- locked test 最终性能。

## 7. 产物

- 发现方案：`docs/plans/route_a_early_complex_sawfm_identifiability_plan_20260725.md`
- 固定确认方案：`docs/plans/route_a_early_complex_sawfm_confirmation_plan_20260725.md`
- 发现程序：`experiments/run_route_a_early_complex_sawfm_identifiability.py`
- 确认程序：`experiments/run_route_a_early_complex_sawfm_confirmation.py`
- 正式发现汇总：`outputs/tapc/route_a_early_complex_sawfm_identifiability/route_a_early_complex_sawfm_identifiability_summary.json`
- 正式确认汇总：`outputs/tapc/route_a_early_complex_sawfm_confirmation/route_a_early_complex_sawfm_confirmation_summary.json`
- 发现汇总 SHA256：`8E95011A83F55185608CF542B1D6C27A7E3D3D40874A90B6B759FEA5C01B258C`
- 确认汇总 SHA256：`94D31A25D9C8A33CE188DBA1DBB7A5C2DF1EAC62E2F8837CCD9CD6B896091FA6`
