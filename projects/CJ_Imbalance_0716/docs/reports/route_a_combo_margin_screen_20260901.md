# Route-A combination-margin screen (2026-09-01)

This development-only candidate adds a bounded combination-level margin to
BCE. For every training sample, the loss compares its true six-label score to
the highest-scoring alternative among the combinations observed in the
training manifest. `margin=0.5`, `weight=0.10`; no validation or locked labels
are used to construct the candidate set.

Strict protocol: label-IF2, 15 held-out combinations, 391 validation samples,
three seeds, fixed F2 backbone and training budget. `locked_test_used=false`;
E5 unchanged.

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.984897 | 0.923616 | 0.718670 | 0.056266 |
| ComboMargin | 0.986864 | 0.915181 | 0.699915 | 0.063797 |

ComboMargin loses exact-combination accuracy by about 1.88 percentage points
and reduces recall for AMA, SinFM and ISRJ. It fails the screen and is stopped.
The result indicates that a margin over observed combinations does not transfer
to the held-out combination geometry; it can instead over-constrain legitimate
multi-label score configurations.

Artifact: [screen summary](../../outputs/route_a_strict_combo_margin_screen_20260901/summary.json)

