# Route-A Logit Adjustment screen (2026-09-01)

The candidate used a fixed prior-to-neutral correction on the six binary logits,
with `bias=-log(p/(1-p))`, `tau=1`, computed only from the training manifest.
The same correction was used during training and evaluation. The protocol was
the strict 15-unseen-combination `label_IF2` development split; no test or
locked-test file was opened and E5 was unchanged.

Three-seed results:

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.984897 | 0.923616 | 0.718670 | 0.056266 |
| LogitAdj | 0.983900 | 0.928895 | 0.719523 | 0.054987 |

The combination gain was only `+0.085 pp` and Macro-AUPR decreased. MFT and
ISRJ FPR increased to about 0.023 and 0.028. The method therefore fails the
promotion gate and is stopped. Prior correction changes the operating point,
but does not solve the underlying unseen-combination boundary.

Artifact: [screen summary](../../outputs/route_a_strict_combo_logitadj_screen_20260901/summary.json)

