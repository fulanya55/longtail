# Route-A frozen F2 feature separability audit (2026-09-05)

## Status

Completed diagnostic; no candidate model was promoted. The first same-day run is explicitly invalid because it evaluated a randomly initialized head; it is retained under outputs/route_a_frozen_f2_feature_separability_20260905/INVALID_RUN.md. This v2 run fully loaded the eight-epoch C1 checkpoint, including classifier weights. No locked test was opened.

## Question

The P3g/P3h failures showed that low-power visibility supervision can damage ISRJ/MFT negative margins. This audit separates two possible causes without changing the deployed model:

1. frozen F2 features do not contain enough class information at low power; or
2. the 30 dB classifier is informative but its scores do not transfer across power domains.

Only train and train-only unseen-identity development views were read. No thresholds were tuned.

## Protocol

The checkpoint is F2_DFMS_Full/dfms_iq, true I/Q [B,2,3000], width 16, feature dimension 128, epoch 8, 257,511 backbone parameters. Features were extracted from 4,096 30 dB fit samples and 256 train-only unseen-identity samples at each 10--50 dB power. Three readouts were compared: the fully loaded F2 logits; a per-label logistic probe trained only on 30 dB frozen features; and a four-fold within-power logistic probe used only to estimate feature-space separability. The probe is diagnostic, not a proposed final model.

## Results

| Power | F2 logits combo | 30 dB linear probe combo | Within-power CV probe combo | F2 AUPR | Within-power AUPR |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.0195 | 0.2812 | 0.3086 | 0.8605 | 0.8743 |
| 15 | 0.0195 | 0.2734 | 0.2852 | 0.8526 | 0.8515 |
| 20 | 0.0156 | 0.3047 | 0.6328 | 0.9371 | 0.9545 |
| 25 | 0.4453 | 0.4102 | 0.9297 | 0.9982 | 0.9989 |
| 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 35 | 0.6523 | 0.6523 | 1.0000 | 0.9982 | 1.0000 |
| 40 | 0.3281 | 0.3242 | 0.9688 | 0.9882 | 0.9998 |
| 45 | 0.2617 | 0.2617 | 0.8008 | 0.9407 | 0.9872 |
| 50 | 0.3242 | 0.3242 | 0.4688 | 0.8933 | 0.9327 |

## Interpretation

The result partitions the problem into two regimes:

* **10--20 dB:** even a within-power readout has low exact-combination accuracy (0.285--0.633). This is evidence of a genuine low-power observability/separability limit. A calibration-only intervention cannot create the missing information.
* **25--40 dB:** within-power probes are almost perfectly separable (0.930--1.000), while the fixed 30 dB head transfers poorly at 25/40 dB. This is primarily a power-domain calibration/representation alignment problem, not a lack of class information.
* **45--50 dB:** within-power separability remains above the frozen F2 cross-power readout, again indicating domain-shifted score geometry; the decline is not explained by a completely absent feature signal.

This explains why P3f improved high-power AUPR but hurt low-power combination decisions, and why P3g/P3h increased ISRJ/MFT false positives: a shared auxiliary objective changes score geometry without supplying the missing 10--20 dB evidence.

## Decision

The audit does not authorize a new expert or locked-test read. It closes the shared-backbone visibility-auxiliary line (P3g/P3h) and provides a sharper design constraint:

* Any low-power innovation must add genuinely new observable information or a physically justified front-end, not just an auxiliary visibility label.
* Any 25--50 dB improvement may use a separately preregistered calibration/alignment diagnostic, but it must preserve the 30 dB anchor and cannot be judged by a single power.
* A final candidate requires simultaneous per-class recall/FPR, exact-combination, low/high power gates, five seeds, and cost audit.

## Artifacts

* Corrected script: experiments/audit_route_a_frozen_f2_feature_separability.py.
* Corrected summary: outputs/route_a_frozen_f2_feature_separability_v2_20260905/frozen_f2_feature_separability_summary.json.
* Role manifest: outputs/route_a_frozen_f2_feature_separability_v2_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_frozen_f2_feature_separability_v2_20260905/terminal_receipt.json.
* Invalid first run notice: outputs/route_a_frozen_f2_feature_separability_20260905/INVALID_RUN.md.
