# Route-A PF plus bounded head hard-negative screen (2026-09-01)

The candidate combined `PFmix25` with a `0.05` top-10%-negative hinge on the
four head labels. It was intended to retain Positive-Focal tail gains while
controlling head false positives. The strict 15-unseen-combination,
label-IF2 development protocol was unchanged; no locked test was read.

Three-seed mean:

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.984897 | 0.923616 | 0.718670 | 0.056266 |
| PF_HN | 0.984450 | 0.923168 | 0.726343 | 0.057545 |

FMB FPR decreased to about 0.080, but MFT FPR increased to about 0.025 and
the macro/Hamming metrics did not improve. The candidate fails the combined
promotion gate and is stopped. A small head-negative hinge is insufficient to
recover the tail/combination gain without introducing another class-specific
tradeoff.

