# Route A development-v3 Focal-72 cosine convergence audit result

## Decision

**The fixed Focal-72 cosine audit does not improve on Focal-36 and does not
approach the frozen H1 linear-BCE baseline.** This closes the remaining
pre-registered one-schedule convergence check for the ordinary Focal loss. It
is a baseline-strength result, not a promotion of B2/F3 and not authorization
to use test-v3.

## Fixed, non-search protocol

- Development-v3 only (SHA-256 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`), seeds `2039--2043`, and the pre-existing role indices, F2 starts and RawSaw SawFM route.
- Focal is fixed at `gamma_pos=0, gamma_neg=3`; 72 unconditional epochs use AdamW classifier/shared learning rates `5e-4/5e-5`, decay `1e-4`, batch 64, clipping 5 and SawFM weight 0.2.
- The only schedule is cosine decay to one percent of the initial rate. There is no early stopping, loss/LR search, calibration alteration, threshold alteration or model selection.
- Platt calibration and per-label FPR<=0.20 thresholds retain their assigned development roles. Locked test, test-v2 and sealed test-v3 flags are all false.

## Convergence evidence

All five runs completed 72 epochs. Best validation epochs are `64, 71, 70,
72, 71`; only 1/5 equals the cap. Across seeds, the last-ten-epoch change is
between `-0.00029` and `+0.00029` for validation macro-AUPR and between
`-0.00117` and `+0.00390` for target weak AUROC. Thus this one fixed schedule
has reached a near plateau, without claiming that every conceivable ordinary
training recipe is exhausted.

## Five-seed result

Focal-72 absolute macro-AUPR / ISRJ AUPR / ISRJ weak recall / weak AUROC /
FPR are `0.89014 / 0.84775 / 0.25827 / 0.56730 / 0.18500`.

| Comparison (first minus second) | ISRJ AUPR | ISRJ weak recall | ISRJ weak AUROC | Macro-AUPR |
| --- | ---: | ---: | ---: | ---: |
| Focal-72 minus Focal-36 | -0.00099 [-0.00284,+0.00083] | -0.00380 [-0.01020,+0.00270] | -0.00056 [-0.00138,+0.00019] | -0.00004 [-0.00050,+0.00049] |
| Focal-72 minus frozen H1 linear BCE | -0.03804 [-0.04852,-0.02954] | -0.06744 [-0.08061,-0.05427] | -0.03539 [-0.05424,-0.01787] | -0.00692 [-0.00988,-0.00391] |

Focal-72 versus Focal-36 is positive in only 2/5 seeds for ISRJ weak recall
and 2/5 for weak AUROC. It is positive in 0/5 seeds for both endpoints versus
H1. The small mean target FPR difference against H1 is `-0.00290`; lower FPR
therefore does not explain the ranking/recall deficit.

## Consequence for the evidence chain

The former open statement, “Focal may simply be under-trained,” is now
substantially narrowed for this fixed and fully archived cosine schedule.
Ordinary-loss convergence does not recover the H1 decoupled-head result.
However, H1 is a stronger classifier baseline than the old F3 periodicity
fusion: the work remains blocked on a **new non-overlapping target-local
evidence mechanism** that can improve MFT/VFT/ISRJ beyond H1 under matched
controls. Neither this result nor H0g/H0h establishes a complete unified
six-label candidate.

Machine-readable output:
`outputs/tapc/route_a_development_v3_focal72_cosine_convergence/route_a_development_v3_focal72_cosine_convergence_summary.json`.
