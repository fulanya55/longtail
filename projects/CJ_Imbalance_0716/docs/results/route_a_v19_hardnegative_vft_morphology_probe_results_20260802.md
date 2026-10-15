# Route A v19: nonlinear VFT morphology hard-negative separability result

## Decision

The v19 identifiability probe **passes**. Nonlinear time-frequency morphology
contains large, stable VFT-vs-ISRJ hard-negative evidence that is absent from
the existing linear ridge diagnostic. This permits a separately preregistered,
fresh-data Q4 integration experiment; it does not itself change Q4.

## Protocol boundary

The source is a new, train-role-only 5,040-identity file, generator seed
`560000000`, SHA-256
`bf5f7aa387d36ccbe7589db52f3c9c74df7f8408ca02260d8631ab9067ec56ee`.
Within ISRJ-present samples, VFT-positive and VFT-absent hard negatives are
balanced in every identity-disjoint role: 768/768 fit, 256/256 early-stop,
and 256/256 held-out evaluation.

The probe receives only 268 deterministic I/Q features: a 16x16 normalized
STFT distribution and 12 frequency-track statistics. The 17,281-parameter
head is `268 -> 64 -> GELU -> dropout(0.10) -> 1`. M-real and M-perm have
identical initialization, architecture and budget. M-perm uses independently
permuted fit and early-stop labels; held-out evaluation always uses real
labels. Neither Q4 logits, predicted width, calibration, nor a decision
threshold is supplied. Locked test, test-v2, test-v3 and any v19 test role
remain unused.

## Held-out five-seed evidence

| Metric | M-real | M-real minus M-perm | Gate |
| --- | ---: | ---: | --- |
| AUROC | 0.9527, CI [0.9515, 0.9539] | +0.4877, CI [+0.4594, +0.5139] | pass |
| AUPR | 0.9614, CI [0.9604, 0.9625] | +0.4816, CI [+0.4566, +0.5065] | pass |
| weak-VFT vs hard-negative AUROC | 0.8416, CI [0.8385, 0.8455] | — | pass |

All five real-minus-permutation AUROC deltas are positive. Parameter count,
role separation, source hash, checkpoint/history/prediction artifacts, and
the test-v3 seal were verified.

## Interpretation

The existing linear VFT ridge score was insufficient because it only gives a
single global score. v19 demonstrates that the distributed 2-D STFT morphology
and temporal track structure contain a nonlinear boundary that distinguishes
true VFT from the physically matched ISRJ confounder, including weak VFT
positives. This is a necessary, not sufficient, condition for correcting Q4:
the next candidate must prove that morphology condition plus predicted width
reduces Q4 false positives without eroding weak/strong VFT recall.

## Artifacts

- Plan: `docs/plans/route_a_v19_hardnegative_vft_morphology_probe_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v19_hardnegative_vft_morphology_probe.m`, `experiments/audit_route_a_v19_hardnegative_vft_morphology_probe.py`, `experiments/run_route_a_v19_hardnegative_vft_morphology_probe.py`
- Formal output: `outputs/tapc/route_a_v19_hardnegative_vft_morphology_probe/`
- Non-evidential smoke: `outputs/tapc/route_a_v19_hardnegative_vft_morphology_probe_smoke/`

