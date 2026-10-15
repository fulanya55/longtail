# Route A 原始 I/Q 物理证据可分性 development-v3 结果

## 1. 结论

三类固定物理摘要中，只有 VFT-STFT 显示明确的原始 I/Q 增量证据：early AUROC `0.64868`，相对 final-temporal `+0.08854`，5/5 种子为正。但其 validation→early 落差 `0.18471` 超过 `0.15` 门槛，因此本轮没有标签正式晋级。MFT-ACF 与 ISRJ-Reset 均显著差于 final-temporal，停止这两个固定摘要方向。

## 2. 五种子结果

| 标签/物理特征 | early AUROC | early AUPR | 相对 final-temporal | validation→early gap | 正增益种子 |
|---|---:|---:|---:|---:|---:|
| MFT-ACF | 0.48017 | 0.18630 | -0.10712 | 0.24871 | 0/5 |
| VFT-STFT | **0.64868** | **0.29130** | **+0.08854** | 0.18471 | **5/5** |
| ISRJ-Reset | 0.49370 | 0.17701 | -0.09891 | 0.15347 | 0/5 |

VFT 的 AUROC 95% CI `[0.62412,0.67483]`，增益 CI `[0.05780,0.11927]`，说明其 early 增益不是偶然；失败仅来自预注册的跨角色泛化门。

## 3. 决策

- VFT：保留 STFT 时频脊机制，下一轮只降低 16×16 图的维数以抑制过拟合；
- MFT：当前自相关固定摘要不再继续调 lag、bin 或 top-k；需要新的可证伪物理机制后才能重启；
- ISRJ：当前相位复位/幅度差分的全局周期摘要不再继续；
- test-v3 仍保持未授权、未使用。

主要产物：`outputs/tapc/route_a_raw_iq_physics_separability_v3/route_a_raw_iq_physics_separability_v3_summary.json`。
