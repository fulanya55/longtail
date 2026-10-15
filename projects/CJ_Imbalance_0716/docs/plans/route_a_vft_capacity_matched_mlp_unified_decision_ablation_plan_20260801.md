# Route A VFT fusion capacity-matched MLP ablation: development-v3 preregistration

Date: 2026-08-01

## Falsifiable question

Does the VFT aligned Ridge-STFT fusion evidence remain useful when its decision
capacity is controlled, or can the observed decision effect be explained by a
small nonlinear head alone?

## Frozen inputs and data roles

- Source decision: the stored `Q2_H1B2_MFTGuard` logits from the completed
  construction run.  The MFT guard, all five non-VFT logits, the H1 backbone,
  and the fit/calibration/early/late roles are frozen.
- VFT evidence: the existing fit-only Ridge-STFT model (`alpha=10`), rebuilt
  from its saved parameters only.  No ridge parameter, feature, or data-role
  selection is allowed.
- Visibility definition: the recorded 10 dB mask.  Fit weak VFT positives have
  sample weight 4; every other fit observation has weight 1.
- `test-v2`, historical locked tests, and sealed `test-v3` are prohibited.
  The runner checks the test-v3 seal and writes all three use flags as false.

## Capacity-matched arms

The shared four VFT inputs are `base_logit`, `ridge_z`,
`uncertainty * ridge_z`, and `tanh(base_logit) * ridge_z`.

| Arm | Input and head | Trainable parameters |
|---|---|---:|
| `Linear6FixedCapacity` | a linear logistic head over the four shared inputs plus fixed `base_logit^2` and `ridge_z^2` statistics | 6 weights + 1 bias = 7 |
| `MLP1CapacityControl` | `Linear(4,1) -> Tanh -> Linear(1,1)` over the four shared inputs | 4 weights + 1 hidden bias + 1 output weight + 1 output bias = 7 |

The two extra linear inputs are deterministic, role-local functions and have
no learned feature extractor.  This deliberately gives the linear arm a
stronger fixed nonlinear-statistics control while exactly matching trainable
parameter count.  The historical Q4 logistic result is reported only as a
source reference, not selected or replaced.

## Training and decision protocol

- Per-arm feature standardization is fit-role only.
- Both heads use full-batch AdamW (`lr=0.01`, `weight_decay=1e-4`), weighted
  BCE logits loss, at most 200 epochs, and patience 20 selected only by the
  existing validation-role VFT AUPR.  Initialization and every operation are
  seed-recorded and deterministic where supported.
- The selected frozen head is independently Platt-calibrated and receives the
  unchanged FPR <= 0.20 threshold procedure through `_evaluate`.
- Five development-v3 seeds (2039--2043) are required; a smoke run validates
  only the code path and never supplies a mechanism conclusion.

## Metrics and interpretation gate

Primary comparison is MLP minus the parameter-matched linear arm for VFT weak
versus negative AUROC.  Secondary metrics are VFT AUPR, weak recall, strong
recall and negative FPR, with paired bootstrap 95% intervals and positive-seed
counts.  The capacity explanation is rejected only if MLP is not better by
0.005 weak AUROC, its AUPR is not higher, and at least four of five seeds are
not positive.  If the MLP is better, capacity remains a live confound; neither
outcome promotes the failed external-OOD candidate or permits test-v3 use.
