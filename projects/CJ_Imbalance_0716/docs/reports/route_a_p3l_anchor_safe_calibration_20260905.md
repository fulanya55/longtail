# Route-A P3l: anchor-safe pooled calibration (2026-09-05)

## Status

**Null effect; not promoted and stopped.** This was a diagnostic-only, development-role experiment. The F2 checkpoint remained frozen and no locked test was opened.

## Design

P3k pooled scalar calibration improved some mid-power combination decisions but reduced the 30 dB anchor. P3l reused exactly that pooled calibrator and applied its residual only when the maximum absolute raw F2 logit was below a preregistered fixed margin of 2.0. High-confidence samples, including the presumed 30 dB anchor, were left unchanged. No margin search was performed.

## Results

| Role | Raw F2 combo | Pooled calibration combo | Anchor-safe combo | Gate fraction |
|---|---:|---:|---:|---:|
| 30 dB validation | 1.0000 | 0.9736 | 1.0000 | 0.000 |
| 10 dB development | 0.2812 | 0.2773 | 0.2812 | 0.000 |
| 15 dB development | 0.2734 | 0.2734 | 0.2734 | 0.000 |
| 20 dB development | 0.3047 | 0.3047 | 0.3047 | 0.000 |
| 25 dB development | 0.4453 | 0.6328 | 0.4453 | 0.000 |
| 35 dB development | 0.6523 | 0.4727 | 0.6523 | 0.000 |
| 40 dB development | 0.3281 | 0.5703 | 0.3281 | 0.000 |
| 45 dB development | 0.2617 | 0.3320 | 0.2617 | 0.000 |
| 50 dB development | 0.3242 | 0.3242 | 0.3242 | 0.000 |

The margin gate never activated because all sampled F2 logits had at least one absolute class logit above 2.0. Consequently, the safe candidate is exactly the raw F2 decision on this role; it cannot claim either calibration gain or a new innovation.

## Decision

Stop P3l without tuning the margin. P3k/P3l together show that pooled calibration can shift mid-power thresholds but either harms the 30 dB anchor or becomes a no-op under a fixed safety gate. Calibration cannot repair the 10--15 dB observability limit. Any future calibration module would require a separately preregistered, power-agnostic confidence/uncertainty gate and five-seed confirmation; it is not authorized by this result.

P3g/P3h visibility auxiliaries, P3i/P3j phase interventions, P4 hard-negative matching, P5 experts, and locked-test access remain stopped.

## Artifacts

* Script: experiments/run_route_a_p3l_anchor_safe_calibration.py.
* Summary: outputs/route_a_p3l_anchor_safe_calibration_20260905/p3l_anchor_safe_calibration_summary.json.
* Role manifest: outputs/route_a_p3l_anchor_safe_calibration_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_p3l_anchor_safe_calibration_20260905/terminal_receipt.json.
