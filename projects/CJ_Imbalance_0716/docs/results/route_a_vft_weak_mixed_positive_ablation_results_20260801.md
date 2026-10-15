# Route A VFT weak-only versus mixed-positive supervision ablation

## Result

On the fixed five-seed development-v3 protocol, replacing strictly weak VFT
positives with all VFT positives during the otherwise identical Ridge-STFT fit
reduces held-out strictly-weak VFT separability:

| Held-out role | Mixed minus weak-only AUROC | 95% bootstrap CI | Mixed minus weak-only AUPR | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| validation | +0.01916 | [-0.05607, +0.09439] | +0.06818 | [-0.02454, +0.15488] |
| early | -0.03265 | [-0.05400, -0.01014] | -0.02264 | [-0.03677, -0.00833] |

The validation result is unstable and therefore cannot be used to select a
supervision regime.  The pre-specified early role is decisive for this
mechanism test: both intervals are below zero.  Strong-positive-dominated
training dilutes the VFT weak-evidence boundary even though it supplies more
positive samples.

## What this establishes—and what it does not

This directly supports the current VFT expert's visibility-selective training
choice.  It does not establish a new candidate, does not retrain a fusion
head, does not choose a threshold, and does not repair the complete
candidate's ISRJ-parameter OOD failure.  It is construction evidence only.

The finding is also useful negatively: ordinary all-positive supervision is
not an acceptable simplification of the VFT weak-component mechanism.  Future
fresh-candidate work must retain a weak-aware objective or demonstrate a
strictly better alternative on a fresh development split.

## Reproducibility

- Plan: [route_a_vft_weak_mixed_positive_ablation_plan_20260801.md](../plans/route_a_vft_weak_mixed_positive_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_weak_mixed_positive_ablation.py`
- Summary: `outputs/tapc/route_a_vft_weak_mixed_positive_ablation/route_a_vft_weak_mixed_positive_ablation_summary.json`
- Five seeds: `2039, 2040, 2041, 2042, 2043`; VFT STFT `16x16`; fit-only standardization; Ridge `alpha=10`; weak threshold `10 dB`.

All records have `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
