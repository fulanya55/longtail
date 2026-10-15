# H1+B2 plus MFT guard and conditional VFT-STFT fusion: development-v3 preregistration

Date: 2026-08-01

## Hypothesis

The validation-independent VFT Ridge-STFT score is not reliable enough as a stand-alone classifier, but it contains sample-aligned physical ridge evidence that can improve the VFT boundary when conditioned on the stronger H1 score. A four-feature logistic fusion head should outperform both a same-capacity H1-only refit head and a role-wise shuffled STFT control.

## Fixed protocol

- Backbone and main head: frozen H1.
- ISRJ: frozen B2 aligned periodicity fusion.
- MFT: frozen peak-count negative guard, residual cap 0.75.
- VFT physical expert: existing fit-only Ridge-STFT, alpha 10, no validation selection.
- VFT fusion features: H1 VFT logit, standardized Ridge-STFT score, uncertainty times score, and tanh(logit) times score.
- VFT fusion: logistic regression, C=0.1, weak-positive sample weight=4.
- Controls: same-capacity base-logit-only refit head and role-wise shuffled STFT fusion head.
- Each complete variant uses the same independent per-label Platt calibration and FPR<=0.20 threshold protocol.
- No hyperparameter search is permitted.

## Variants

- Q0: H1.
- Q1: H1+B2.
- Q2: Q1 plus frozen MFT guard.
- Q3: Q2 plus same-capacity VFT base-only refit head.
- Q4: Q2 plus aligned VFT Ridge-STFT fusion.
- Q5: Q2 plus shuffled VFT Ridge-STFT fusion.

## VFT mechanism gates: Q4

1. VFT weak-vs-negative AUROC gain over Q2 >= 0.010.
2. VFT weak recall gain over Q2 >= 0.010.
3. VFT AUPR does not decline.
4. VFT weak AUROC gain over Q3 >= 0.005.
5. VFT weak AUROC gain over Q5 >= 0.005.
6. At least four of five seeds improve VFT weak AUROC over Q2 and Q5.
7. Mean VFT FPR <= 0.20.
8. Non-VFT logits are exactly preserved between Q2 and Q4.

## Unified Goal gates: Q4 versus Q0

The original 13 promotion requirements are evaluated unchanged. Passing development-v3 is only a construction result. The complete candidate, all five VFT heads, calibration rules, thresholds, MFT experts, and code must then be frozen and applied without refitting to a fresh identity-isolated development set. test-v3 remains sealed.
