# Route A MFT residual-cap removal unified-decision ablation

## Question and hypothesis

The historical MFT cap study used the repeatedly consumed construction-v3
endpoint. It therefore cannot establish the requested core ablation on the
current Q4 six-label decision protocol. This study directly tests whether the
extra `0.75` residual scale is required once the frozen Q4 decision, its
independent calibration role, and its independent FPR-constrained threshold
role are all retained.

The tested contrast is deliberately narrow. `cap=1.0` removes only the extra
`0.75` hard scale; the negative-only direction, uncertainty factor, and
`tanh` natural bound stay in place. It is not an unbounded residual experiment.

## Fixed design

| Arm | MFT logit | Other five label logits | Calibration and threshold roles |
|---|---|---|---|
| Q4_Cap075 | stored Q4 aligned decision (`0.75 * uncertainty * min(tanh(e), 0)`) | stored, unchanged | arm-specific Platt calibration on `calibration`; weak-recall/FPR<=0.20 threshold on `threshold` |
| Q4_NaturalCap100 | reconstruct Q1 MFT with `1.0 * uncertainty * min(tanh(e), 0)` | exactly copied from Q4_Cap075 | same role assignment, refit only inside each arm |

- Development-v3 SHA-256 must equal
  `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`.
- Use identity-isolated seeds `2039, 2040, 2041, 2042, 2043` and the same
  weak definition (`10 dB`) for analysis only.
- Reconstruct the `0.75` arm from stored Q1 plus frozen aligned MFT evidence
  and require max absolute MFT discrepancy from stored Q4 <= `1e-6` for every
  role and seed.
- No expert, backbone, VFT head, calibration data, threshold data, training,
  candidate selection, test-v2, locked-test, or test-v3 access is allowed.

## Measurements and decision rules

Report Q4_NaturalCap100 minus Q4_Cap075 with a bootstrap 95% CI over five
identity-isolated seeds for MFT weak-vs-negative AUROC, weak recall, AUPR,
negative FPR, weak-vs-order>=4-negative AUROC, and order>=4-negative FPR.
Also report macro AUPR/F1 and exact preservation of every non-MFT logit.

This is a mechanism and safety ablation, not a candidate-promotion test. The
protocol-integrity gate requires: sealed test-v3, exact reconstruction,
non-MFT preservation, and each arm's MFT negative FPR <=0.20 after its own
pre-assigned role procedure. Regardless of effect direction, a completed,
safe contrast supplies the current-protocol core-14 evidence for item 8. The
effect may only be described as a candidate-change opportunity if a separate,
fresh mechanism and external-OOD protocol is later registered.
