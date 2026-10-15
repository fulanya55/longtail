# Route A unified visibility experts: v11 failure, shared-policy repair, and v12 confirmation

## Evidence boundary

- Backbone, H1 classifier, RawSaw branch, ISRJ periodic expert, MFT peak-count
  expert, VFT Ridge-STFT expert, fusion heads, and residual cap were frozen.
- development-v11 was a fresh frozen confirmation for candidate v1.  After the
  MFT mechanism gate failed, v11 was downgraded to diagnostic reuse and was not
  treated as independent evidence for the repair.
- Candidate v2 was frozen before development-v12 feature extraction.
- development-v12 contains 25,200 train-only samples with unique
  `receiver_noise_seed` values in `322677743--322702942`.  Its SHA-256 is
  `358e08e6eecbf2c866a137ebeed847475ce1ea2efe2c399b6789a0a8a5bbffe9`.
  Candidate v2 was frozen before this line evaluated it, but the exact file had
  already been consumed by the FC-PTV research line.  It is thus a line-held-out
  frozen confirmation, not a project-globally fresh confirmation; a changed
  candidate requires development-v13.
- test-v3 remained sealed and unused throughout.

## Candidate v1 on development-v11

The VFT mechanism gate passed 9/9 and the unified promotion gate passed 13/13,
but the MFT hard-negative mechanism gate failed.  Relative to H1+B2, the MFT
guard improved MFT AUPR by `0.00415`, weak recall by `0.02857`, and weak-vs-
negative AUROC by `0.01656`; however, high-order hard-negative FPR increased by
`0.00108` and order-5 FPR increased by `0.00750`.

The failure was not caused by the physical score losing ranking information:
hard-negative AUROC increased by `0.01254`.  The cause was decision-policy
coupling.  The residual is non-positive, but candidate v1 fitted a separate MFT
Platt calibration and threshold on development-v3.  On v11, that separately
frozen decision mapping reversed the intended no-new-positive property.

## Fixed repair

Candidate v2 keeps every learned component and hyperparameter unchanged.  It
uses Q1 H1+B2 calibration and threshold for MFT, and Q4 calibration and
thresholds for all other labels.  With a monotone calibration and a non-positive
MFT residual, this gives a sample-wise structural invariant: an H1+B2 MFT
negative decision cannot become positive because of the guard.

The construction-v3 and diagnostic-v11 audit passed all preregistered checks.
High-order hard-negative FPR changed by `-0.02833` on v3 and `-0.02825` on v11;
order-5 FPR changed by `-0.02500` and `-0.03150`; MFT weak-vs-negative AUROC
changed by `+0.01861` and `+0.01656`.  The MFT weak-recall cost versus H1+B2 was
bounded at `-0.00956` and `-0.01196`.  Non-MFT outputs and SawFM outputs were
exactly unchanged.

## Frozen line-held-out development-v12 confirmation

Candidate v2 manifest:

- name: `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2`
- manifest SHA-256:
  `c29ce67bd0384cf6d54095343773b557258e287394c89f8deb3acbd6ba943747`
- refit or hyperparameter search on v12: none

### VFT mechanism gate: 9/9 passed

- weak-vs-negative AUROC versus Q2: `+0.01706`
- weak recall versus Q2: `+0.01871`
- AUPR versus Q2: `+0.01337`
- weak AUROC versus same-capacity base-only control: `+0.01706`
- weak AUROC versus shuffled expert: `+0.01757`
- VFT negative FPR: `0.19810`
- all five seeds had positive aligned-expert gains over Q2 and shuffled control

### MFT shared-policy gate: 8/8 passed

- high-order hard-negative FPR: `-0.02658`, bootstrap 95% CI
  `[-0.03100, -0.02217]`
- order-5 hard-negative FPR: `-0.02800`, bootstrap 95% CI
  `[-0.03850, -0.01950]`
- hard weak-positive-vs-negative AUROC: `+0.01182`, bootstrap 95% CI
  `[+0.01039, +0.01293]`
- MFT weak-recall cost versus H1+B2: `-0.01265`, bootstrap 95% CI
  `[-0.01524, -0.01035]`
- all five seeds reduced hard-negative FPR
- negative-to-positive MFT flips: `0`
- non-MFT probability and prediction mismatch: `0`

### Unified promotion gate: 13/13 passed

Mean candidate-v2 change relative to H1:

| Metric | Delta or value |
|---|---:|
| six-label macro-AUPR | `+0.00660` |
| six-label macro-F1 | `+0.01099` |
| MFT/VFT/ISRJ mean weak recall | `+0.03759` |
| MFT/VFT/ISRJ mean weak-vs-negative AUROC | `+0.03886` |
| target mean negative FPR | `0.18606` |
| target mean strong recall | `+0.02354` |
| MFT AUPR | `+0.00414` |
| VFT AUPR | `+0.01337` |
| ISRJ AUPR | `+0.02211` |
| SawFM AUPR | exactly unchanged |
| SawFM weak recall | exactly unchanged |
| SawFM-present macro-F1 | `+0.01049` |
| SawFM-absent macro-F1 | `+0.01134` |

All five seeds improved the principal target weak-recall endpoint.  The lowest
per-seed target weak-recall gain was `+0.02949`; the preregistered gate concerns
the five-seed mean, which was `+0.03759`.

## Current conclusion

The supported mechanism is not merely a larger shared network.  Three
label-specific physical/temporal evidence routes improve their intended labels,
VFT aligned evidence beats capacity and shuffled controls, MFT has a formal
no-new-positive safety property, and the line-held-out v12 result preserves the
RawSaw SawFM output exactly.  This is strong development evidence, not
project-globally fresh or final locked-test evidence.

Before test-v3, the final candidate still requires paired significance and
effect sizes, full stratified robustness, external/OOD confirmation of the
complete candidate, engineering-cost accounting, exact ablation coverage, and
the unresolved latest multi-label long-tail baseline audit.
