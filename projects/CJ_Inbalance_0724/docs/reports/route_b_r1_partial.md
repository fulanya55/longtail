# Route B R1 baseline — partial development screen

Status: **partial**. One seed (`2041`), one epoch, and 20/10 samples per label combination were completed for the three parameter-head paths. This is a code/data-path and directional screen only; it is not the preregistered multi-seed performance result.

- Development split: generated from the six-class training MAT with split seed `20260826`.
- Variants: `legacy_mlcnn_global`, `legacy_mlcnn_class_conditioned`, `legacy_mlcnn_class_conditioned_detached`.
- B0 classification-only and B4 scheduled warm-up remain protocol entries but have not received a performance run in this stage.
- No test-v2 was created or opened; locked test was not used.
- The exact rows are in `docs/results/route_b_r1_baseline.csv`; the run receipt is `docs/receipts/route_b_r1_result_receipt.json`.

The next required action is full development training with the frozen R1 protocol, at least five seeds, unchanged split/loss/optimizer/threshold/budget, followed by per-parameter and paired-power audits.
