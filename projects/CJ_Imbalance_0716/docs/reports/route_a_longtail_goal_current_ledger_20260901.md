# Route-A long-tail Goal current ledger (2026-09-01)

All entries are development-v2. No test/locked-test file was opened and E5 was
not modified.

## BCE boundary

The old P3 nominal IF=5/CF=5 manifest actually had IF≈1.37--1.51 and
CF≈12.25--23.50. The corrected fixed-budget protocol identifies `label_IF2`
as the first repeatable BCE bottleneck. Five-seed normal validation gives
Macro-AUPR 0.996153, Macro-F1 0.940715 and exact-combination accuracy 0.765332.
The strict 15-unseen-combination split gives BCE exact-combination accuracy
0.628133.

## Candidate ledger

| candidate | seeds | strict unseen-combo result | status |
|---|---:|---|---|
| Class-Balanced BCE | 3 | exact-combo 0.719523; effectively BCE | stop |
| ASL | 3 | exact-combo 0.609548; severe FPR increase | stop |
| Positive-Focal | 5 | exact-combo 0.718159 vs BCE 0.628133; FMB FPR +0.0285 | mechanism only |
| DRW | 3 | no gain over BCE on normal validation | stop |
| ComboSampler | 3 | lower combo/Hamming performance | stop |
| TailPositiveFocal | 3 | lower than BCE | stop |
| PFmix25 | 5 | combo +0.46 pp; Macro-AUPR slightly lower | stop |
| LogitAdj | 3 | combo +0.085 pp; Macro-AUPR lower | stop |
| PF_HN | 3 | combo +0.77 pp; MFT FPR higher | stop |
| ComboMargin | 3 | combo -1.88 pp | stop |

Positive-Focal is the only method that improves unseen-combination recall by a
large amount on the first fresh split. On a second independent split it still
improves combination accuracy (55.80%→59.40%) and Hamming loss (12.11%→10.49%),
but ISRJ negative-FPR increases by 0.84 percentage points. With the preregistered
per-label FPR cap of 0.5 percentage points, it remains a development candidate,
not a promoted final method. No final method has been promoted.

The two fresh strict-split confirmations are recorded in
`route_a_newsplit_strict_positivefocal_confirm_20260901.md` and
`route_a_newsplit2_strict_longtail_gate_20260901.md`. PF_HN was screened on the
second split and stopped: combination accuracy gained only 0.18 percentage
points while Macro-F1 and Hamming worsened.

## Mixed-long-tail protocol

Joint integer-programming feasibility checks show that exact independent
IF=2 and CF=4/8 cannot be imposed at fixed budget on the current 63-combination
source. This is recorded as a protocol limitation; no nominal mixed condition
is claimed without achieved IF/CF evidence.

## Cost

All candidates use the same F2 model with 257,511 parameters and approximately
4.45G convolution/linear MACs per sample. Typical per-run elapsed time is about
9--10 seconds on the RTX 5080 for the strict 8-epoch, 1000-sample screen. The
loss-only candidates therefore do not change model parameter cost; their
differences are decision geometry and error allocation.

## Next controlled direction

The current loss family is exhausted for this data geometry. The next valid
development experiment is a fresh-split, pre-registered objective that combines
a bounded combination constraint with an explicit head-FPR cap, followed by a
new feasibility audit for mixed-long-tail data. Locked-test access remains
prohibited until explicit authorization.

Machine-readable aggregate: [route_a_longtail_goal_aggregate_20260901/summary.json](../../outputs/route_a_longtail_goal_aggregate_20260901/summary.json)

BCE boundary figure and table: [bce_boundary.png](../../outputs/route_a_longtail_goal_aggregate_20260901/bce_boundary.png), [bce_boundary.csv](../../outputs/route_a_longtail_goal_aggregate_20260901/bce_boundary.csv)
