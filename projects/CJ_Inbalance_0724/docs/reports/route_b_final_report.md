# Route B final evidence report (development-only, gate-audited)

## 1. Research question

Test whether conditional parameter heads improve masked estimation for the six fixed classes while preserving the recognition task under the fixed [N,2,3000] I/Q contract.

## 2. Data roles, identity split and hashes

R0 verified the active six-class train MAT, finite labels/targets and masks, and unique identity grouping. The development split is seed 20260826 with 22,680 train and 2,520 validation identities. The historic 0716 locked test was not opened. Nine power views were treated as paired views sharing receiver_noise_seed, never as independent samples. See the manifest receipt for current code/config/data-role hashes.

## 3. Models

B0 is classification-only; B1 uses a global joint parameter head; B2 is class-conditioned; B3 adds stop-gradient class probabilities; B4 warms the parameter loss for two epochs. Physical-FiLM (26 features) and PSD spectral-FiLM (19 features) are implemented and smoke-verified only; they are not performance-promoted.

## 4. Training protocol

All R1 variants used the same split, AdamW (lr 3e-4, weight decay 1e-4), masked SmoothL1 parameter loss, parameter weight 0.1, threshold 0.5, batch size 64, width 16, head dimension 256, hidden dimension 64, 12 epochs maximum, patience 4, and seeds 2041--2045. Aggregation uses deterministic percentile bootstrap 95% CIs over development seeds.

## 5. Validity and masking

A parameter contributes only when label=1, parameter_mask=1 and target is finite. Inactive NaNs were not zero-filled; normalization statistics were fitted only on valid training targets.

## 6. R1 classification and parameter results

| Variant | Macro-AUPR mean | Macro-F1 mean | Parameter Macro-NMAE mean | Gate status |
|---|---:|---:|---:|---|
| B0 | 0.98844 | 0.94593 | NA | reference |
| B1 | 0.98697 | 0.94214 | 0.57680 | reference |
| B2 | 0.98739 | 0.94204 | 0.60849 | stopped: parameter gate failed |
| B3 | 0.98689 | 0.94245 | 0.62627 | stopped: parameter gate failed |
| B4 | 0.98698 | 0.94310 | 0.62722 | stopped: parameter gate failed |

B0 Macro-AUPR=0.98844 and Macro-F1=0.94593. B1 parameter Macro-NMAE=0.57680. B2/B3/B4 preserved the preregistered classification safety limits, but their parameter relative improvements versus B1 were -5.49%, -8.57% and -8.78%, respectively. The 5% parameter-improvement hard gate therefore failed for all three conditional branches. The stop receipt records that no threshold retuning, resampling, model enlargement or test feedback was used.

## 7. Per-class parameter evidence

| Variant | Class | Oracle NMAE | End-to-end NMAE |
|---|---|---:|---:|
| B1 | FMB | 0.47136 | 0.47352 |
| B1 | ISRJ | 0.54934 | 0.53130 |
| B1 | MFT | 0.66858 | 0.66514 |
| B1 | SawFM | 0.61309 | 0.60942 |
| B1 | SinFM | 0.48969 | 0.47761 |
| B1 | VFT | 0.66868 | 0.66812 |
| B2 | FMB | 0.49294 | 0.49511 |
| B2 | ISRJ | 0.61877 | 0.59556 |
| B2 | MFT | 0.70449 | 0.70008 |
| B2 | SawFM | 0.65288 | 0.65302 |
| B2 | SinFM | 0.48500 | 0.46926 |
| B2 | VFT | 0.69684 | 0.68928 |
| B3 | FMB | 0.50042 | 0.50183 |
| B3 | ISRJ | 0.64585 | 0.62649 |
| B3 | MFT | 0.71584 | 0.71131 |
| B3 | SawFM | 0.67535 | 0.67779 |
| B3 | SinFM | 0.49835 | 0.48528 |
| B3 | VFT | 0.72172 | 0.71481 |
| B4 | FMB | 0.49582 | 0.49717 |
| B4 | ISRJ | 0.62642 | 0.60729 |
| B4 | MFT | 0.74285 | 0.73940 |
| B4 | SawFM | 0.67458 | 0.67612 |
| B4 | SinFM | 0.49761 | 0.48436 |
| B4 | VFT | 0.72593 | 0.72066 |

## 8. Visibility and low-power evidence

Nominal validation visibility summaries (corrected denominator: two parameters per class per identity) are:

| Variant | Oracle coverage | Predicted gate rate | Safe-stop rate on negatives |
|---|---:|---:|---:|
| B1 | 0.50794 | 0.47340 | 0.49206 |
| B2 | 0.50794 | 0.47284 | 0.49206 |
| B3 | 0.50794 | 0.47487 | 0.49206 |
| B4 | 0.50794 | 0.47431 | 0.49206 |

These are development validation quantities only, not an open-world or locked-test claim.

## 9. Paired-power consistency

The same validation identities were evaluated over JNR 10--50 dB train views. Mean values below are per variant and JNR; they are descriptive paired-view evidence, not independent-sample confidence intervals.

| Variant | JNR dB | Macro-AUPR | Macro-F1 | Parameter Macro-NMAE |
|---|---:|---:|---:|---:|
| B1 | 10 | 0.92777 | 0.73651 | 0.75564 |
| B1 | 15 | 0.96317 | 0.87881 | 0.65794 |
| B1 | 20 | 0.98256 | 0.93140 | 0.73602 |
| B1 | 25 | 0.99274 | 0.95313 | 0.83997 |
| B1 | 30 | 0.99636 | 0.96679 | 1.01233 |
| B1 | 35 | 0.99686 | 0.97315 | 1.25590 |
| B1 | 40 | 0.99608 | 0.97459 | 1.54669 |
| B1 | 45 | 0.99232 | 0.96192 | 1.84859 |
| B1 | 50 | 0.97713 | 0.85827 | 2.13495 |
| B2 | 10 | 0.92749 | 0.73630 | 0.81320 |
| B2 | 15 | 0.96394 | 0.88002 | 0.70587 |
| B2 | 20 | 0.98365 | 0.93497 | 0.79142 |
| B2 | 25 | 0.99330 | 0.95469 | 0.87630 |
| B2 | 30 | 0.99662 | 0.96822 | 1.02364 |
| B2 | 35 | 0.99704 | 0.97386 | 1.26874 |
| B2 | 40 | 0.99607 | 0.97402 | 1.55788 |
| B2 | 45 | 0.99163 | 0.95485 | 1.83446 |
| B2 | 50 | 0.97582 | 0.84262 | 2.08403 |
| B3 | 10 | 0.92605 | 0.73603 | 0.82486 |
| B3 | 15 | 0.96284 | 0.88206 | 0.69530 |
| B3 | 20 | 0.98248 | 0.93440 | 0.77857 |
| B3 | 25 | 0.99251 | 0.95409 | 0.87258 |
| B3 | 30 | 0.99620 | 0.96682 | 1.03388 |
| B3 | 35 | 0.99684 | 0.97221 | 1.29476 |
| B3 | 40 | 0.99612 | 0.97293 | 1.59587 |
| B3 | 45 | 0.99239 | 0.96109 | 1.87467 |
| B3 | 50 | 0.97846 | 0.85944 | 2.12673 |
| B4 | 10 | 0.92573 | 0.72343 | 0.81644 |
| B4 | 15 | 0.96232 | 0.87031 | 0.72333 |
| B4 | 20 | 0.98261 | 0.93337 | 0.80928 |
| B4 | 25 | 0.99290 | 0.95479 | 0.89290 |
| B4 | 30 | 0.99648 | 0.96783 | 1.03752 |
| B4 | 35 | 0.99695 | 0.97213 | 1.28889 |
| B4 | 40 | 0.99597 | 0.97231 | 1.57979 |
| B4 | 45 | 0.99206 | 0.95986 | 1.85082 |
| B4 | 50 | 0.97778 | 0.86495 | 2.09642 |

## 10. Multi-seed and resource evidence

All five variants have five completed development seeds. Resource measurements are recorded separately in route_b_resource_metrics.csv and are approximate inference benchmarks, not dataset throughput claims.

## 11. Failures and stop branches

R1 B2, B3 and B4 are stopped because the parameter hard gate failed. R4's ten strict ablations were preregistered and preflight-smoke-verified but not executed because their conditional-head prerequisites were not met. R2 physical/spectral branches remain proposal/smoke evidence only. No locked test or test-v2 result is claimed.

## 12. Literature correspondence

The literature-to-experiment plan maps communication/radar/emitter/jamming parameter-extraction findings to the shared I/Q backbone, spectral/physical auxiliary hypotheses, conditional masking and paired-power checks. Each mapping remains a hypothesis until its matched experiment passes the preregistered gate.

## 13. Claimable conclusions

- R0 data, identity split, mask semantics, code-path and smoke readiness are verified.
- B1 is the only retained joint baseline from the completed development gate; B2/B3/B4 are not promoted.
- Train-only paired-power and visibility tables are complete with locked-test usage false.

## 14. Not claimable

No final-model superiority, conditional-head gain, low-power robustness, open-world generalization, independent test-v2 performance, or R4 ablation conclusion is claimable.

## 15. Next minimum experiment

Design a fresh preregistered parameter-head hypothesis or revise the objective, then rerun identity-isolated development-v2 with the same hard-gate discipline before any independent test-v2 evaluation.
