# Route-A P3i: normalized complex phase-increment input screen (2026-09-05)

## Status

**Failed the main-anchor gate; stopped.** This is development-only evidence. The F2 backbone, BCE protocol, split, threshold, and optimizer were unchanged; no locked test was opened.

## Hypothesis

For modulation classes whose identity is encoded in phase evolution, replace normalized I/Q with the adjacent complex product, represented as cos(delta-phase) and sin(delta-phase). This tests whether a physically interpretable phase-increment representation adds low-power information without changing F2_DFMS_Full/dfms_iq.

## Protocol

Both variants used F2_DFMS_Full/dfms_iq, width 16, input shape [B,2,3000], 4,096 30 dB fit samples, 2,048 held-out 30 dB validation samples, 256 train-only unseen-identity development samples per 10--50 dB power, BCEWithLogits, AdamW lr 1e-4, weight decay 1e-4, batch 16, two epochs, gradient clip 5.0, seeds 2039/2040/2041, threshold 0.5.

## Results

| Variant | 30 dB Macro-AUPR | 30 dB Macro-F1 | 30 dB exact combo |
|---|---:|---:|---:|
| Raw normalized I/Q | 0.99999950 | 0.99970764 | 0.99820964 |
| Phase increment | 0.99225777 | 0.96224842 | 0.80875651 |
| Difference | -0.00774172 | -0.03745922 | -0.18945313 |

The phase-increment representation reduces 30 dB combination accuracy in every seed (0.80127, 0.83350, 0.79150). It also reduces 10--25 dB development AUPR, so it does not address the low-power observability limit. It improves some high-power metrics (40--50 dB combination or AUPR), but those gains cannot compensate for the main-anchor collapse.

## Failure signature and interpretation

The candidate increases errors across several labels, especially ISRJ and MFT; for example, seed 2039 has MFT FPR 0.205 and ISRJ recall 0.829 at 30 dB. The adjacent phase product is noise-sensitive at low SNR and discards amplitude/Cartesian evidence that F2 uses for FMB, MFT, and ISRJ. The result is evidence that phase evolution is complementary at high power but cannot replace raw normalized I/Q in the shared F2 classifier.

## Decision

Stop P3i. Do not tune the phase representation or combine it as an unbounded second branch in this line; that would change more than the preregistered single-input intervention. A future complementary residual would require a new preregistration and explicit parameter/cost/anchor gates. P3g/P3h visibility auxiliaries, P4 hard-negative matching, P5 experts, and locked-test access remain stopped.

## Artifacts

* Script: experiments/run_route_a_p3i_phase_increment_frontend.py.
* Summary: outputs/route_a_p3i_phase_increment_20260905/p3i_phase_increment_summary.json.
* Statistics: outputs/route_a_p3i_phase_increment_20260905/p3i_phase_increment_statistics.json.
* Role manifest: outputs/route_a_p3i_phase_increment_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_p3i_phase_increment_20260905/terminal_receipt.json.
