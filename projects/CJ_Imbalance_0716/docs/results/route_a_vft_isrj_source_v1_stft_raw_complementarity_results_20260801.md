# VFT--ISRJ source-v1: STFT plus raw-source complementarity result

## Result

The raw phase/copy/ridge/reset descriptor contains genuine phase-organised
information, but it does not add enough discrimination beyond the full 16x16
STFT grid to justify a new VFT expert or a changed unified candidate.

| Early held-out comparison | Delta AUROC, 95% bootstrap CI | Delta AUPR, 95% bootstrap CI |
|---|---:|---:|
| STFT + RawSource minus STFT + PhaseScrambled | +0.01187, [+0.00593, +0.01854] | +0.02529, [+0.01590, +0.03840] |
| STFT + RawSource minus STFT | +0.00445, [+0.00104, +0.00892] | +0.01282, [+0.00254, +0.02263] |

The raw-augmented arm has early AUROC/AUPR `0.66378 / 0.54072`, compared with
`0.65191 / 0.51543` for its equal-dimensional phase-scrambled control and
`0.65934 / 0.52791` for STFT alone.  Its validation-to-early AUROC gap is
`0.00666`, so the conclusion is not explained by a fit-adjacent validation
gain.

## Pre-registered decision

The source-level phase-information control succeeds: RawSource augmentation
beats the phase-scrambled equal-capacity control by at least `+0.010` in both
early AUROC and AUPR.  However, the candidate fails the more important
incremental gate against the already strong STFT reference: its early AUROC
gain is only `+0.00445` and only one of five identity-isolated replications
reaches `+0.010`.

Therefore this route is stopped.  The result establishes that the hand-built
features are not random extra capacity, but also establishes that the complete
STFT grid has already absorbed nearly all of their useful source-separation
information.  It must not be appended to the frozen VFT expert, used to
produce development-v13, or cited as an OOD repair.

## Scope and reproducibility

- Plan: [route_a_vft_isrj_source_v1_stft_raw_complementarity_plan_20260801.md](../plans/route_a_vft_isrj_source_v1_stft_raw_complementarity_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_isrj_source_v1_stft_raw_complementarity.py`
- Summary: `outputs/tapc/route_a_vft_isrj_source_v1_stft_raw_complementarity/route_a_vft_isrj_source_v1_stft_raw_complementarity_summary.json`
- Source data SHA-256: `c6e041cf286a1edc26477b4595963d0802ee990201cee05e617b3001008f2cd5`.
- Five source identity partitions: `2039, 2040, 2041, 2042, 2043`.

This is fresh-source, fixed-feature source-identifiability evidence only.  No
calibration, threshold, fusion, external OOD data, historical locked test,
test-v2, or test-v3 was used; all saved records state
`locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false`.
