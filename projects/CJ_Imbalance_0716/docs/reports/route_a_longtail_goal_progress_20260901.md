# Route-A long-tail Goal progress (2026-09-01)

All results below are development-v2 only. `locked_test_used=false` throughout
this stage and the frozen E5 artifact was not modified.

## Boundary and protocol

- Six classes: FMB, AMA, SawFM, SinFM, MFT, ISRJ.
- Backbone: F2_DFMS_Full / `dfms_iq`, real I/Q `[B,2,3000]`.
- Old P3 nominal IF=5/CF=5 was audited as effective IF≈1.37--1.51 and
  CF≈12.25--23.50.
- New fixed-budget manifests use 1600 samples and audited label-only
  IF≈1/2/3 or combo-only CF≈1/4/8.
- `label_IF2` is the first stable BCE bottleneck: five-seed exact-combination
  accuracy 0.765332 and Macro-F1 0.940715, versus about 0.999 combination
  accuracy for the balanced anchor.

## Method evidence

| candidate | protocol | evidence | decision |
|---|---|---|---|
| Class-Balanced BCE | label_IF2, normal validation and strict unseen-combo screen | near-zero change | stopped |
| ASL | label_IF2, strict unseen-combo screen | exact-combo 0.609548; severe FPR increase | stopped |
| Positive-Focal | label_IF2, strict unseen-combo, five seeds | exact-combo 0.718159 vs BCE 0.628133; FMB FPR +0.0285 | mechanism evidence only |
| DRW BCE→CB | label_IF2, three seeds | exact-combo 0.823568 vs BCE 0.825358 on normal validation | stopped |
| ComboSampler | label_IF2, three seeds | exact-combo 0.816406; Hamming worse | stopped |
| TailPositiveFocal | strict unseen-combo, three seeds | exact-combo lower than BCE | stopped |
| PFmix25 | strict unseen-combo, five seeds | exact-combo +0.46 pp; Macro-AUPR slightly lower; tail FPR higher | stopped |
| Logit Adjustment | strict unseen-combo, three seeds | exact-combo +0.085 pp; Macro-AUPR lower; MFT/ISRJ FPR higher | stopped |
| PF_HN | strict unseen-combo, three seeds | FMB FPR lower but MFT FPR higher and macro metrics unchanged | stopped |

## Mixed-long-tail feasibility

Joint integer-programming audits show that exact IF=2 and CF=4/8 cannot be
simultaneously enforced at fixed budget on the current 63-combination source.
The obstruction is the observed label/combination co-occurrence geometry, not a
training implementation failure. An emergent mixed manifest may be reported as
diagnostic only; it must use achieved IF/CF rather than requested values.

## Current scientific conclusion

Global or class-frequency reweighting can increase tail responses, but it does
not reliably improve the coupled tail/combination boundary. Positive-Focal is
useful as evidence that tail/combination recall can be improved, but its head
false-positive cost prevents promotion. The present bottleneck is a joint
decision-geometry problem under sparse combination coverage.

## Next gate

1. Freeze the completed negative results and do not tune on the consumed strict
   confirmation split.
2. Use a fresh development-v2 split to test one explicitly bounded
   combination-level boundary objective with a head-FPR cap.
3. In parallel, implement Logit Adjustment only if its definition is kept as a
   pre-registered operating-point correction rather than a post-hoc threshold
   search.
4. Revisit mixed-long-tail only with a jointly generated, feasibility-audited
   source; do not relabel the current infeasible construction as IF=2/CF=4.
5. No locked-test action is permitted without new explicit authorization.

