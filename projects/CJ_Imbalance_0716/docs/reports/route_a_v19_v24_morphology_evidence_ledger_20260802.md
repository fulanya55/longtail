# Route A v19-v24 morphology/visibility evidence ledger

## Scope and decision boundary

This ledger separates a completed **VFT-versus-ISRJ morphology identifiability observation** from five subsequent failed attempts to convert it into a safe visibility-aware correction. It does not claim a unified six-label VTSE-VBF model, does not alter frozen Q4, and does not authorize test-v3.

All v19-v24 sources are independently generated train-only sets with unique `receiver_noise_seed` identities and disjoint fit/early/evaluation roles. All formal receipts declare `locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false`.

## Evidence by falsifiable mechanism

| Stage | Fresh-source mechanism | Formal result | Scientific decision |
| --- | --- | --- | --- |
| v19 | 16x16 STFT morphology MLP versus same-capacity label permutation | Real expert AUROC `0.9527`, AUPR `0.9614`; weak VFT versus hard-negative AUROC `0.8416` | The feature has nonlinear discriminative information; this is an identifiability result, not a fusion result |
| v20 | Frozen predicted-width × morphology residual | Reduces FPR, but weak recall `-0.0162`; morphology-only has better FPR | Reject multiplicative width condition |
| v21 | Weak-recall-constrained morphology residual coefficient | Q4 FPR `-0.0344`, weak recall `-0.0159`; candidate costs `+0.0555` FPR versus global-only | Reject downstream coefficient selection |
| v22 | Equal weak/strong positive loss mass | weak AUROC `-0.00778`, CI `[-0.01351,-0.00199]` versus BCE | Reject static positive-mass reweighting |
| v23 | BCE plus weak-positive/hard-negative ranking term | weak AUROC `-0.00097`, CI `[-0.00219,+0.00026]` | Reject pair-ranking loss on the fixed feature |
| v24 | 32x32 local STFT concatenation versus equal-dimension projected-base control | weak AUROC `-0.04457`, CI `[-0.05847,-0.03295]`; overall AUROC `-0.01740` | Reject direct multi-resolution concatenation |

## What has been established

1. The VFT/ISRJ hard-negative problem has a real morphology signal, so the failure is not a zero-information task.
2. Weak-positive protection is a distinct safety requirement: aggregate VFT recall or FPR can improve while weak VFT recall degrades.
3. On the current frozen morphology feature, neither weak-positive loss mass nor a direct ranking penalty improves held-out weak ordering.
4. More STFT pixels are not automatically useful. A high-resolution local image coupled to the same small classifier is worse even against a parameter- and input-dimension-matched no-new-information control.
5. Therefore no v19-v24 candidate can be represented as a validated VTSE-VBF fusion improvement; Q4 and RawSaw remain frozen historical evidence only.

## Excluded directions

- Re-searching residual coefficient, threshold, or calibration values on v20-v24.
- Repeating width multiplication, weak-positive mass weighting, pair-ranking coefficient tuning, or direct STFT concatenation.
- Attaching any v19-v24 morphology expert to Q4 or evaluating any of them on test-v3.

## Next evidence gate

Before a further network change, the project needs a source-audited architecture review that distinguishes a label-conditioned *local evidence branch* from the already rejected frozen-token query and naive feature concatenation. A future candidate must include an explicit untouched global safety bypass and an information/parameter-matched ablation. It must first pass a fresh identity-isolated expert-level test for MFT, VFT, and ISRJ, then move to a separate six-label end-to-end development protocol with fair mainstream-loss baselines and full ablations.

## Linked formal records

- `docs/results/route_a_v19_hardnegative_vft_morphology_probe_results_20260802.md`
- `docs/results/route_a_v20_frozen_dual_condition_residual_results_20260802.md`
- `docs/results/route_a_v21_visibility_balanced_morphology_selector_results_20260802.md`
- `docs/results/route_a_v22_visibility_balanced_morphology_expert_results_20260802.md`
- `docs/results/route_a_v23_weak_pair_ranking_morphology_expert_results_20260802.md`
- `docs/results/route_a_v24_multiresolution_weak_evidence_probe_results_20260802.md`
