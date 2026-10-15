# Route A P2: full-target observability protocol

Status: preregistered. P2 is a distinct project-local external physics
protocol, not a P1 retry and not an original-0716/test-v3 role.

## Single hypothesis

An all-target external protocol can satisfy an H0 physical-evidence gate when
each target is given an independently justified observable aperture: P1's
already specified non-overlapping MFT copies and coherent high-Doppler VFT,
plus the **previously independently verified** H0g/H0h/H0s ISRJ condition of
high replica count and aligned periodicity. The ISRJ choice is fixed from those
pre-P1 evidence records, not selected from the failed P1 envelope-
autocorrelation values.

## Fixed differences from P1

| Item | P1 (closed) | P2 (new protocol) | Rationale |
| --- | --- | --- | --- |
| MFT | 120/230/340 us copies | unchanged | P1 H0 passed with known-chirp control. |
| VFT | 20--60 kHz, six coherent pulses | unchanged | P1 H0 passed with timestamp-permutation control. |
| ISRJ replicas | 1--5 | fixed 9 | H0s verified high-replica aligned-periodicity evidence. |
| ISRJ slice width | 5--20 us | uniform 25--30 us | H0s parameter interval. |
| ISRJ H0 statistic | generic envelope ACF, lags 50--1,000 | RMS-normalized joint complex/envelope periodicity maximum at lags 200--600 | fixed low-dimensional counterpart of the previously validated aligned-periodicity feature family. |
| ISRJ control | global sample permutation | 25-sample fixed block permutation | preserves local waveform values but destroys the long aligned periodic structure, matching the existing B2 control principle. |
| identity blocks | P1 `4,200,000,000` / `4,250,000,000` | P2 `4,150,000,000` / `4,160,000,000` | new, non-overlapping designated P2 blocks. |

Every P2 identity has absent, weak 8 dB and strong 20 dB views. Non-target
components, powers, sample seed and receiver noise are exactly shared within
the triplet. The formal H0 has 900 identities / 2,700 views, 300 identities per
target. It creates no train/test role.

## Fixed H0 and gate

MFT and VFT use P1's unchanged true scores and controls. ISRJ uses the average
of (a) the maximum absolute complex ACF and (b) the maximum envelope ACF over
lags 200--600 after RMS normalization; the fixed control block-permutes 25
sample blocks by `numpy.random.default_rng(20260803)`. No score uses labels,
arm, target parameter, power, seed or prior logits while it is calculated.

For all three targets, weak-vs-absent true AUROC must be >=0.75, true-control
AUROC >=+0.10 with pair-bootstrap lower CI >0, paired weak>absent fraction
>=0.65, and strong AUROC must be no worse than weak AUROC -0.02. All targets
must pass. Failure closes P2 with no variation of copies, PRI, Doppler, replica
count, width, lags, block size, grid, score, loss, model, calibration or
threshold. Passing H0 would authorize only the design of a new five-seed
equal-capacity C0/C3/C4 protocol; it never authorizes test-v3.
