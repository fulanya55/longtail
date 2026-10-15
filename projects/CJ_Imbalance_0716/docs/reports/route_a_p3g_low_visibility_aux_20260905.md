# Route-A P3g: low-power visibility auxiliary screen (2026-09-05)

## Status

**Failed main-anchor stability; stopped.** This is development-only evidence, not a promoted candidate. locked_test_used=false and test_files_opened=false; no expert module was built.

## Hypothesis and implementation

P3f showed that low/high consistency-gradient conflict is real but PCGrad shifts capacity toward high power. P3g tests a different, bounded hypothesis: use the paired 10 dB view to supervise a six-output visibility head through the generator-provided weak_component_mask, while the classification loss remains BCE on the fixed 30 dB anchor. The auxiliary head shares the F2 feature extractor but is not a power-dependent inference gate and does not change the classifier threshold.

The target is defined only for active selected labels; weak_component_mask < 0.5 is the visible target, following the existing P2c convention. Baseline and candidate use the same AuxModel parameterization so the comparison does not conflate an extra head with training loss.

## Fixed protocol

* Backbone: F2_DFMS_Full / dfms_iq, real I/Q [B,2,3000], width 16.
* Fit: 4,096 paired 30/10 dB samples; validation: 2,048 held-out 30 dB samples.
* Development: train-only unseen receiver identities across 10--50 dB, 256 samples per power.
* Loss: 30 dB BCE; candidate adds 10 dB visibility BCE with aux_weight=0.1.
* AdamW (lr=1e-4, weight_decay=1e-4), batch 16, 2 epochs, clip 5.0, seeds 2039/2040/2041, CUDA cuda:0, threshold 0.5.

## Three-seed validation results

| Variant | Macro-AUPR | Macro-F1 | Exact combination |
|---|---:|---:|---:|
| F2 BCE | 0.99999990 | 0.99933453 | 0.99593099 |
| BCE + low-power visibility | 0.99999935 | 0.99059887 | 0.93994141 |
| Difference | -0.00000055 | -0.00873566 | -0.05598958 |

The candidate is unstable: seed 2039 reaches 0.99463 exact-combination accuracy, but seeds 2040 and 2041 fall to 0.92139 and 0.90381. AUPR remains near one while exact-combination and F1 collapse, indicating a threshold/logit calibration failure rather than a lack of ranking capacity.

## Failure signature and power development

The collapse is concentrated in false positives for ISRJ and MFT. At 30 dB, seed 2040 has ISRJ FPR 0.1301 and MFT FPR 0.0309; seed 2041 has ISRJ FPR 0.1111 and MFT FPR 0.0689, while recall remains near one. The low-power auxiliary target therefore pushes shared features/logit offsets in a way that destroys negative-class separation on the 30 dB anchor.

The candidate improves low-power AUPR at 15/20/25 dB and high-power AUPR at 45/50 dB, but exact-combination accuracy falls at 10/15/20 dB, 30 dB, 35 dB, 40 dB, 45 dB and 50 dB on average. It fails the non-negotiable requirement of preserving the 30 dB anchor while improving low-power robustness.

## Cost audit

The candidate adds 1,030 parameters (258,541 versus the 257,511-parameter F2 backbone). Its post-warmup median training time is approximately 25.00 s/seed versus 16.16 s/seed for the BCE control (about 1.55x); the 43.97 s baseline measurement for seed 2039 is a startup outlier. Inference backbone latency is unchanged in kind, but the training cost is higher.

## Decision

Stop P3g and retain the complete failure evidence. The root cause is mechanism-level interference: a low-power visibility target is not equivalent to the 30 dB class-discriminative target, and unrestricted shared-backbone gradients alter ISRJ/MFT negative margins. This is not fixed by more epochs or threshold retuning within the same split. Do not promote the auxiliary head, do not build P5 experts, and do not read locked test.

## Reproducibility artifacts

* Script: experiments/run_route_a_p3g_low_visibility_aux.py (SHA-256 in terminal receipt).
* Summary: outputs/route_a_p3g_low_visibility_aux_20260905/p3g_low_visibility_aux_summary.json.
* Statistics: outputs/route_a_p3g_low_visibility_aux_20260905/p3g_low_visibility_aux_statistics.json.
* Role manifest: outputs/route_a_p3g_low_visibility_aux_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_p3g_low_visibility_aux_20260905/terminal_receipt.json.

## Next bounded direction

The next candidate, if continued, must protect the 30 dB classification gradient explicitly (for example, a detached/frozen-feature calibration probe or an anchor-projected auxiliary gradient). It must be screened first on three seeds with an explicit gate: no 30 dB exact-combination/F1 loss, no increase in ISRJ/MFT FPR, and simultaneous low/high-power improvement. Otherwise the mechanism line is stopped rather than tuned.
