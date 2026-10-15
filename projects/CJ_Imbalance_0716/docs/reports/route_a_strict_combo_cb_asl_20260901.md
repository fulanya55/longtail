# Route-A strict-combination CB-BCE and ASL screen (2026-09-01)

Three-seed screen on the strict `label_IF2`/15-unseen-combination protocol.
No locked-test access; E5 unchanged.

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| CB-BCE | 0.984916 | 0.923833 | 0.719523 | 0.056124 |
| ASL | 0.995042 | 0.917499 | 0.609548 | 0.080563 |

CB-BCE is effectively indistinguishable from BCE. ASL has high Macro-AUPR but
loses 11 percentage points of exact-combination accuracy and produces severe
false positives (mean FPR approximately 0.27 FMB, 0.15 SawFM, 0.21 MFT and
0.11 ISRJ). Both candidates fail the promotion gate and are stopped.

Artifact: [screen summary](../../outputs/route_a_strict_combo_cb_asl_screen_20260901/summary.json)

