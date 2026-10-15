# Route-A P3f: PCGrad paired-power consistency screen (2026-09-05)

## Status

Exploratory **partial support; not promoted and not frozen**. This is a development-only result. `locked_test_used=false` and `test_files_opened=false`; no expert module was constructed.

## Motivation and hypothesis

P3c--P3e showed a conflict between low-power (10--25 dB) and high-power (35--50 dB) consistency objectives: improving one end often sacrificed the other. P3f tests a narrowly defined mechanism hypothesis: calculate the low/high consistency gradients separately and apply PCGrad projection only when their dot product is negative, then add the projected consistency gradient to the 30 dB BCE gradient. The backbone, input, split, optimizer, threshold, and consistency coefficient are unchanged.

## Fixed protocol

* Backbone: F2_DFMS_Full / `dfms_iq`, true I/Q input `[B,2,3000]`, width 16, 257,511 trainable parameters.
* Fit: 4,096 paired samples from 30 dB; validation: 2,048 held-out 30 dB samples.
* Paired development views: 10, 15, 20, 25, 30, 35, 40, 45, 50 dB, using train-only unseen `receiver_noise_seed` identities.
* Loss: 30 dB BCE-with-logits plus `lambda=0.1` consistency. Baseline is BCE only; candidate uses low/high PCGrad projection.
* AdamW (`lr=1e-4`, `weight_decay=1e-4`), batch 16, 2 epochs, gradient clipping 5.0, seeds 2039/2040/2041, CUDA `cuda:0`.
* Threshold 0.5; metrics include macro-AUPR, macro-F1, exact combination accuracy, per-label recall/FPR, and per-power development metrics.

## Results (three-seed means)

| Variant | 30 dB macro-AUPR | 30 dB macro-F1 | 30 dB exact combo |
|---|---:|---:|---:|
| F2 BCE | 0.99999859 | 0.99909088 | 0.99446615 |
| BCE + PCGrad consistency | 0.99999907 | 0.99911926 | 0.99462891 |
| Difference | +0.00000048 | +0.00002838 | +0.00016276 |

The paired validation differences are not significant at this exploratory sample size (Wilcoxon p=0.18--0.32; see `p3f_pcgrad_statistics.json`).

## Power-generalization result

PCGrad improves the high-power combination metric at 35 dB (+2.60 percentage points) and 40 dB (+1.95 pp), and high-power AUPR at 45 dB (+0.95 pp) and 50 dB (+0.44 pp). It simultaneously decreases low-power AUPR at 10 dB (-0.28 pp), 20 dB (-0.36 pp), and 25 dB (-0.09 pp), with 25 dB combination accuracy -1.04 pp. Thus it does not satisfy the predeclared requirement of improving both low and high power without harming the 30 dB anchor.

The gradient diagnostic confirms the proposed conflict exists: negative low/high gradient dot products occur in roughly 35--41% of mini-batches, while mean cosine is close to zero (about 0.008--0.028). Removing the conflict changes the allocation of capacity but does not recover low-power separability.

## Cost audit

Both variants have 257,511 parameters and the same single-sample CUDA latency profile. PCGrad takes about 45.62 s per seed versus 25.12 s for BCE (1.82x wall-clock), due to three autograd passes and projection. This is an optimization cost, not a model-capacity gain.

## Decision and failure analysis

Decision: **partial support only; stop PCGrad tuning on this split**. The result supports the diagnosis that the low/high objectives conflict, but it does not support PCGrad as a robust Route-A innovation. The remaining failure is representational/observability-limited low-power separability rather than merely gradient interference; the 30 dB anchor is already saturated, and high-power gains are offset by low-power losses. Do not build P5 experts based on this result, and do not read locked test.

## Reproducibility artifacts

* Script: `experiments/run_route_a_p3c_power_consistency.py` (SHA-256 recorded in terminal receipt; `pcgrad` mode).
* Summary: `outputs/route_a_p3f_pcgrad_20260905/p3c_power_consistency_summary.json`.
* Statistics: `outputs/route_a_p3f_pcgrad_20260905/p3f_pcgrad_statistics.json`.
* Role manifest: `outputs/route_a_p3f_pcgrad_20260905/role_manifest.json`.
* Terminal receipt: `outputs/route_a_p3f_pcgrad_20260905/terminal_receipt.json`.

## Next bounded experiment

Keep P4 matched-hard-negative and P5 expert construction stopped. The next permitted candidate should target low-power observability with an independently specified bounded mechanism (for example, an auxiliary low-power visibility/readout objective or a frozen low-power calibration probe), screened first on three seeds and promoted only after five seeds, confidence intervals, per-class/per-power gates, and cost audit. Any candidate that trades low-power recall/AUPR for high-power gains is retained as a failure boundary rather than tuned further.
