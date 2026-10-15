# Route A VFT ISRJ-OOD: frozen-threshold transport diagnostic

Date: 2026-08-01

## Question

The frozen Q4 VFT Ridge-STFT decision improves external OOD ranking but has
high frozen-threshold FPR. This read-only diagnostic asks whether the failure
is plausibly correctable by threshold transport alone, or whether weak-VFT
versus negative ranking is itself insufficient.

## Fixed inputs and boundary

- Use only already-written external OOD Q4 predictions, probabilities,
  thresholds, labels and weak masks for seeds 2039--2043.
- No model, calibrator, threshold, loss, feature, or external refit is written.
- An oracle threshold is calculated only as a post-hoc diagnostic: the lowest
  probability threshold that respects OOD negative FPR <=0.20. It is never
  eligible for selection, deployment, fresh-development construction, or
  test-v3 use.

## Predefined interpretation

For each seed report frozen and oracle FPR/weak recall, oracle threshold shift,
and frozen weak-vs-negative AUROC. Ranking is called insufficient when mean
AUROC <0.65. Threshold-only correction is rejected if ranking is insufficient
or if oracle FPR control loses more than 0.03 weak recall on average. These
are causal-diagnosis conditions, not performance gates.
