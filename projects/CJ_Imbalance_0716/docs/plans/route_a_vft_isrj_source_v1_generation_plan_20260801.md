# Fresh VFT-ISRJ source-v1 diagnostic data plan

## Why a new source diagnostic is necessary

The five development-v3 label-LT fit/validation roles contain no exact
co-label-matched weak-VFT versus VFT-absent/ISRJ-present cases.  Relaxing that
match would make a source-evidence result indistinguishable from a co-label
shortcut.  The already-consumed ISRJ parameter-OOD-v1 dataset cannot be reused
to choose a new physical descriptor.  This plan therefore creates a fresh,
train-only, identity-disjoint source diagnostic.

## Generator and identity contract

- Controlled six-label generator, `balanced` protocol, 63 non-empty label
  patterns x 80 samples = 5,040 train-only samples.
- Base sample seed: `154000000`; expected sample range
  `[154000001, 154005040]` and receiver range `[352677743, 352682782]`.
- These ranges are disjoint from documented development-v2 through v14,
  test-v2 through test-v14 contracts, the sealed test-v3 range, and ISRJ
  parameter-OOD-v1.
- Output: `C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_source_v1`.
- No test split is generated, opened, or reserved; this is a source mechanism
  diagnostic, not a final evaluation set.

## Five fixed source-role replications

For each seed `2039`--`2043`, shuffle identities **within each exact six-label
combination** and allocate 48 fit / 16 validation / 16 early identities.  Thus
each of the 63 patterns appears in every role and no receiver identity crosses
roles within a seed.  The source classifier is trained only on fit, has no
calibration or threshold selection, and reports validation and early only.

## Required audit gates before a physical feature run

1. MAT is train-named, has 5,040 unique receiver identities, and exact
   registered receiver range;
2. every non-empty combination has exactly 80 samples;
3. all five role manifests are receiver-disjoint and cover all samples once;
4. every role has nonzero exact-context-matched weak-VFT positives and
   VFT-absent/ISRJ-present negatives.

If any condition fails, source-feature fitting does not begin.  `test-v3` is
never accessed by generation, role construction, audit, or diagnostic fitting.
