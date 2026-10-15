# Route-A frozen F2 scalar-logit calibration audit (2026-09-05)

## Status

Completed diagnostic only. The F2 model was frozen; no checkpoint, threshold, or locked-test file was changed or read. Calibration probes were fit/evaluated only on train-only unseen-identity development views.

## Question

The frozen-feature audit showed that 25--50 dB features remain separable while the fixed 30 dB head transfers poorly, whereas 10--20 dB feature separability is intrinsically weaker. This audit asks whether the cross-power error can be explained by a scalar per-label logit calibration alone.

## Protocol

The full C1 F2_DFMS_Full/dfms_iq checkpoint (including classifier head) was used. A per-label scalar logistic calibration was (a) fit on the 30 dB fit role and applied to every power, or (b) fit/evaluated by four-fold cross-validation within each train-only development power. The latter is an upper-bound diagnostic, not a deployable final model. The fixed 0.5 decision threshold was used after calibration; no locked test was opened.

## Results

| Power | Raw F2 combo | 30 dB-fitted scalar calibration | Within-power CV scalar calibration |
|---:|---:|---:|---:|
| 10 | 0.2812 | 0.2812 | 0.2695 |
| 15 | 0.2734 | 0.2734 | 0.2812 |
| 20 | 0.3047 | 0.3047 | 0.5352 |
| 25 | 0.4453 | 0.4375 | 0.9258 |
| 30 | 1.0000 | 1.0000 | 1.0000 |
| 35 | 0.6523 | 0.6641 | 0.9102 |
| 40 | 0.3281 | 0.3281 | 0.7656 |
| 45 | 0.2617 | 0.2617 | 0.4766 |
| 50 | 0.3242 | 0.3242 | 0.3320 |

Macro-AUPR is nearly unchanged by scalar calibration because ranking is unchanged. The large combination gains at 20--45 dB under within-power calibration therefore come from score geometry/threshold alignment, not new feature information. The absence of recovery at 10/15 dB (and instability at 50 dB) confirms that calibration alone cannot solve the low-power limit.

## Decision

This audit supports a two-regime Route-A design constraint:

1. 10--15 dB requires a physically justified observation/front-end improvement; a shared visibility loss or scalar calibration is insufficient.
2. 20--45 dB may benefit from a separately preregistered power-domain calibration/alignment module, but it must be evaluated without sacrificing the fixed 30 dB anchor and with per-class FPR gates.

No calibration module is promoted from this audit because the development calibration uses power labels and is only an upper bound. P3g/P3h shared visibility mechanisms remain stopped; P4/P5 and locked-test access remain stopped.

## Reproducibility artifacts

* Script: experiments/audit_route_a_frozen_f2_logit_calibration.py.
* Summary: outputs/route_a_frozen_f2_logit_calibration_20260905/frozen_f2_logit_calibration_summary.json.
* Role manifest: outputs/route_a_frozen_f2_logit_calibration_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_frozen_f2_logit_calibration_20260905/terminal_receipt.json.
* The first print-only NameError was fixed before the successful rerun; no model/data computation changed.
