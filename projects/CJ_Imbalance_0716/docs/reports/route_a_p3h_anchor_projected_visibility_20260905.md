# Route-A P3h: anchor-projected low-power visibility auxiliary (2026-09-05)

## Status

**Failed main-anchor stability; stopped.** This is development-only evidence. No locked test was opened and no expert module was constructed.

## Hypothesis

P3g showed that direct shared-backbone low-power visibility supervision destabilized the 30 dB classifier. P3h therefore projects the 10 dB visibility gradient away from the 30 dB BCE gradient whenever their dot product is negative, then applies the projected auxiliary gradient. The backbone, data roles, loss weight, threshold, and split are otherwise fixed. This is an anchor-protection mechanism, not an inference-time expert.

## Protocol

F2_DFMS_Full/dfms_iq, true I/Q [B,2,3000], width 16; 4,096 30/10 dB paired fit samples; 2,048 held-out 30 dB validation samples; 256 train-only unseen-identity development samples at each 10--50 dB power; AdamW lr 1e-4, weight decay 1e-4, batch 16, two epochs, clip 5.0; seeds 2039/2040/2041; auxiliary weight 0.1; CUDA cuda:0; threshold 0.5.

## Results

| Variant | Macro-AUPR | Macro-F1 | Exact combination |
|---|---:|---:|---:|
| F2 BCE control | 0.99999985 | 0.99930760 | 0.99576823 |
| Anchor-projected visibility | 0.99999940 | 0.99023059 | 0.93733724 |
| Difference | -0.00000045 | -0.00907701 | -0.05843099 |

Seed-level exact-combination accuracy for the candidate is 0.99512, 0.91699, and 0.89990. The candidate's AUPR remains near one, but F1 and exact-combination collapse due to false positives, not ranking failure.

## Diagnostics and failure cause

The anchor and visibility gradients conflict in 74--96% of mini-batches, with mean cosine between -0.048 and -0.131. Projection removes the first-order opposing component, but the shared representation still drifts across updates and changes the ISRJ/MFT negative margins. At 30 dB, ISRJ FPR is 0.1421/0.1121 and MFT FPR is 0.0279/0.0778 for seeds 2040/2041. Low-power AUPR rises at 10--25 dB and 45--50 dB, but exact-combination accuracy decreases at nearly every power, including the 30 dB anchor.

This demonstrates that gradient conflict is not the only failure mode: even an anchor-projected auxiliary target is not aligned with class-discriminative negative margins. The low-power visibility label is therefore not a safe shared representation objective for this six-class task.

## Cost and decision

The model has 258,541 parameters (1,030 auxiliary-head parameters above F2). Candidate median training time is 34.19 s/seed versus 16.23 s/seed for the control, about 2.11x. Decision: stop P3h and the shared low-power visibility auxiliary line. Do not tune its weight, do not promote it, do not build P5 experts, and do not read locked test.

## Artifacts

* Script: experiments/run_route_a_p3g_low_visibility_aux.py, invoked with aux_mode=anchor_projected.
* Summary: outputs/route_a_p3h_anchor_projected_visibility_20260905/p3g_low_visibility_aux_summary.json.
* Statistics: outputs/route_a_p3h_anchor_projected_visibility_20260905/p3h_anchor_projected_statistics.json.
* Role manifest: outputs/route_a_p3h_anchor_projected_visibility_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_p3h_anchor_projected_visibility_20260905/terminal_receipt.json.

## Next direction

The shared-backbone visibility auxiliary mechanism has now failed in both direct and anchor-projected forms. Any further Route-A work must avoid injecting this target into the classifier representation. A bounded next option is a frozen-feature calibration/readout probe used only diagnostically; it may be promoted only if it improves low-power decisions without changing the frozen 30 dB classifier. Otherwise retain the negative result and keep P4/P5/locked-test boundaries unchanged.
