# Route-A long-tail follow-up: tail-only focal screen (2026-09-01)

This was a three-seed development screen on the same strict unseen-combination `label_IF2` protocol. `locked_test_used=false`; E5 remained frozen.

The candidate applied focal modulation only to the two least frequent labels (MFT and ISRJ), while retaining BCE for head labels. The intent was to preserve the head-class FPR safety that global Positive-Focal lost.

| method | Macro-AUPR | exact combination |
|---|---:|---:|
| BCE | approximately 0.9845 | approximately 0.720 |
| TailPositiveFocal | approximately 0.9805 | approximately 0.623 |

Per-seed combination results were:

- seed 2039: 1.0000 vs 1.0000;
- seed 2040: 0.5882 vs 0.4373;
- seed 2041: 0.5729 vs 0.4322.

Tail-only focal therefore failed the screen and is stopped. Modulating only positive terms of the tail labels did not preserve the global combination boundary; the method is not promoted.

The next candidate should explicitly control head false positives while preserving the global BCE negative geometry, for example a bounded BCE/Positive-Focal interpolation or a combination-level margin applied only to verified hard negatives. Any such candidate must be screened on a fresh development split or pre-registered holdout before confirmation.

