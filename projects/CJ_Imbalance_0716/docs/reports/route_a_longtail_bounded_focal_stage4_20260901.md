# Route-A Stage 4: bounded focal interpolation (2026-09-01)

Status: development-v2 completed; no locked-test access and E5 unchanged.

After global Positive-Focal improved unseen-combination recall but increased head FPR, a pre-registered bounded interpolation was screened:

`L = 0.75 * BCE + 0.25 * PositiveFocal(gamma=2)`.

The strict protocol used the same 15 unseen combinations, 391 validation samples, 1000 training samples, effective label IF=2, and five paired seeds.

## Three-seed screen

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.984822 | 0.923661 | 0.719523 | 0.056266 |
| PFmix25 | 0.985717 | 0.926758 | 0.727195 | 0.054703 |

FMB FPR decreased in the screen and the candidate passed to two additional seeds.

## Five-seed confirmation

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.971306 | 0.894404 | 0.630179 | 0.075448 |
| PFmix25 | 0.971122 | 0.895963 | 0.634783 | 0.074425 |

The combination gain was only `+0.460 pp`, Macro-AUPR decreased slightly, and MFT/ISRJ FPR increased (about `+0.0019` and `+0.0039` on average). The candidate therefore fails the confirmation gate and is stopped.

## Mechanism

The interpolation reduces the severe FMB FPR increase of global Positive-Focal, but the reduced focal weight is not sufficient to produce a reliable unseen-combination gain. This supports the interpretation that the unresolved problem is a coupled tail-recall/head-boundary tradeoff, not a single global loss-weight setting.

Artifacts:

- [Three-seed screen](../../outputs/route_a_strict_combo_pf_mix25_screen_20260901/summary.json)
- [Two-seed confirmation](../../outputs/route_a_strict_combo_pf_mix25_confirm_20260901/summary.json)
- [Runner](../../experiments/run_route_a_bce_boundary_screen.py)

