# Route-A P3k: pooled cross-power logit calibration (2026-09-05)

## Status

Completed diagnostic; **not promoted**. The F2 backbone and classifier were frozen, no final threshold was tuned, and no locked-test file was opened.

## Design

The full C1 F2_DFMS_Full/dfms_iq checkpoint was used. Twelve scalar parameters (one affine logistic probe per selected label) were fitted on paired train-role logits pooled from 10, 15, 20, 25, 30, 35, 40, 45 and 50 dB. The purpose was to test whether cross-power combination errors could be corrected without changing representation or adding an expert. Development evaluation used train-only unseen receiver identities.

## Results

| Role | Raw F2 Macro-AUPR | Raw exact combo | Pooled calibration Macro-AUPR | Pooled calibration exact combo |
|---|---:|---:|---:|---:|
| 30 dB held-out validation | 1.0000 | 1.0000 | 1.0000 | 0.9736 |
| 10 dB development | 0.8605 | 0.2812 | 0.8605 | 0.2773 |
| 15 dB development | 0.8526 | 0.2734 | 0.8526 | 0.2734 |
| 20 dB development | 0.9371 | 0.3047 | 0.9371 | 0.3047 |
| 25 dB development | 0.9982 | 0.4453 | 0.9982 | 0.6328 |
| 35 dB development | 0.9982 | 0.6523 | 0.9982 | 0.4727 |
| 40 dB development | 0.9882 | 0.3281 | 0.9882 | 0.5703 |
| 45 dB development | 0.9407 | 0.2617 | 0.9407 | 0.3320 |
| 50 dB development | 0.8933 | 0.3242 | 0.8933 | 0.3242 |

The monotonic scalar transform leaves AUPR unchanged. It improves 25/40/45 dB combination decisions but reduces the 30 dB anchor from 1.0000 to 0.9736 and reduces 35 dB. It does not improve 10--20 dB.

## Decision

P3k is a **calibration-only partial result, not a valid final candidate**. It confirms that some mid-power errors are score-geometry/threshold alignment, but a single pooled calibration cannot preserve the 30 dB anchor and improve all powers. More importantly, it cannot create missing 10--15 dB information. Do not fit it on locked test, and do not tune its affine parameters on this split.

Combined with the frozen feature audit, the evidence supports two separate research questions: a physical observation problem at 10--15 dB and a conditional calibration/alignment problem at 20--50 dB. Any future calibration candidate must have an explicit anchor-preservation constraint and be evaluated with per-class FPR gates.

## Artifacts

* Script: experiments/run_route_a_p3k_pooled_logit_calibration.py.
* Summary: outputs/route_a_p3k_pooled_logit_calibration_20260905/p3k_pooled_logit_calibration_summary.json.
* Role manifest: outputs/route_a_p3k_pooled_logit_calibration_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_p3k_pooled_logit_calibration_20260905/terminal_receipt.json.
