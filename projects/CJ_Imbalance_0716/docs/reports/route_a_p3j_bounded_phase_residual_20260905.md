# Route-A P3j: bounded phase residual screen (2026-09-05)

## Status

**Null increment; innovation gate failed and the line is stopped.** The raw I/Q path remained the main F2 classifier. No locked test was opened and no expert was constructed.

## Hypothesis and protocol

P3i showed that replacing raw I/Q by phase increments discards important Cartesian/amplitude evidence. P3j retained raw normalized I/Q and added a small phase-increment residual branch. The residual logit was bounded by 0.1*tanh(scale) with scale=0 initialization. Both control and candidate used the complete C1 checkpoint (including classifier), the same 30 dB fit/validation split, 10--50 dB train-only unseen-identity development views, BCE, AdamW (1e-4/1e-4), batch 16, two epochs, threshold 0.5, and seeds 2039/2040/2041.

## Results

| Variant | 30 dB Macro-AUPR | 30 dB Macro-F1 | 30 dB exact combo |
|---|---:|---:|---:|
| F2 raw-I/Q control | 1.00000000 | 0.99994694 | 0.99967448 |
| F2 + bounded phase residual | 1.00000000 | 0.99997343 | 0.99983724 |
| Difference | 0 | +0.00002649 | +0.00016276 |

The apparent gain is a single 1/2048 decision in one seed and is not significant (Wilcoxon p=0.32). The residual scale after two epochs is -0.000077, -0.000226, and +0.000298, corresponding to an effective residual multiplier below 3e-5. The candidate therefore behaved as the control, not as an activated complementary mechanism.

## Power development

Low-power combination accuracy was unchanged at 10/15/20 dB and improved only 0.26 percentage points at 25 dB. AUPR changes were within small stochastic fluctuations and were negative at 15, 20, 25, 35, 40, 45 and 50 dB on average. The candidate did not address the 10--20 dB observability limit. At 35 dB combination accuracy decreased 0.65 percentage points.

## Decision and implication

P3j is retained as a **null-increment failure boundary**, not a promoted candidate. The bounded residual did not receive enough gradient signal to activate under the fixed 30 dB anchor, and activating it would require changing the bound, initialization, loss, or power training protocol, which would constitute a new experiment rather than a continuation of this preregistered screen. Do not tune those quantities on the same split. Raw normalized I/Q remains the accepted F2 input.

The sequence P3g/P3h (shared visibility), P3i (phase replacement), and P3j (bounded phase residual) now provides convergent negative evidence against low-power improvements that do not add new physical observations. P4/P5 and locked-test access remain stopped.

## Artifacts

* Script: experiments/run_route_a_p3j_bounded_phase_residual.py.
* Summary: outputs/route_a_p3j_bounded_phase_residual_20260905/p3j_bounded_phase_residual_summary.json.
* Statistics: outputs/route_a_p3j_bounded_phase_residual_20260905/p3j_bounded_phase_residual_statistics.json.
* Role manifest: outputs/route_a_p3j_bounded_phase_residual_20260905/role_manifest.json.
* Terminal receipt: outputs/route_a_p3j_bounded_phase_residual_20260905/terminal_receipt.json.
