# Route A v18: fresh-data bounded predicted-width VFT residual

## Single falsifiable hypothesis

The v16/v17 chain showed that predicted ISRJ width is available from I/Q and
aligned with fixed-Q4 VFT false-positive risk. **H-v18** is stricter: on fresh
identity-isolated data, a bounded downward VFT-logit residual driven by the
frozen predicted-width score will reduce `VFT=0, ISRJ=1` false positives more
than a same-capacity ISRJ-only residual, without materially reducing VFT
positive recall.

## New data and roles

- Generate a new train-role-only six-label source with base seed `540000000`,
  5,040 unique receiver identities, ISRJ width uniform in `[10,30] us`, and
  replicas `[3,4]`. No test file is generated.
- Hash and audit the source before inference. Split identities deterministically
  by `receiver_noise_seed mod 10`: fit `{0..5}`, selection `{6,7}`, and held
  out development evaluation `{8,9}`. No identity crosses roles.
- v17 (including its predictions) is not read by this runner. Locked test,
  test-v2, and sealed test-v3 remain forbidden.

## Fixed models and candidate rule

For each of the five frozen Q4 classifier seeds, replay the immutable
`H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` path. Replay the matching frozen
v16 width probe and its fit-only normalizer. There is no backbone, expert,
calibration, or threshold refit.

Let `z_vft` be Q4's original VFT logit, `p_isrj` its original calibrated ISRJ
probability, and `w_hat` the frozen width prediction in seconds. Define the
fixed bounded width evidence:

`e_width = clamp((w_hat - 20 us) / 10 us, 0, 1) * p_isrj`.

The candidate is `z'_vft = z_vft - c * e_width`; all other logits are exact
Q4 logits. The same-capacity non-width control is
`z'_vft = z_vft - c * p_isrj`. Both have exactly one fitted scalar `c`; Q4's
existing VFT calibrator and threshold are then applied unchanged. The width
center and scale come from the v16 width-training support, not v17 outcomes.

For each seed and rule, select `c` only on the fit role from the preregistered
grid `{0.0, 0.1, ..., 1.0}`: maximize VFT hard-negative FPR reduction subject
to VFT-positive recall loss <= `0.010`; ties choose the smallest `c`. The
selection role never changes `c`; it only applies the same safety screen.

## Endpoints and gates

Primary endpoint: VFT FPR on held-out `VFT=0 AND ISRJ=1` identities. Safety
endpoint: held-out VFT positive recall. Also report VFT AUPR, VFT weak recall,
ISRJ metrics, SawFM metrics, per-seed values, and fixed-threshold identity
checks.

The candidate advances only if, on held-out development identities:

1. Candidate minus Q4 hard-negative VFT FPR <= `-0.030`, with >=4/5 negative
   seed deltas.
2. Candidate minus same-capacity non-width control <= `-0.010`, with >=4/5
   negative seed deltas.
3. Candidate VFT positive-recall loss is no worse than `-0.010` in mean and
   every seed.
4. Candidate VFT AUPR drop is no more than `0.010`; SawFM AUPR/weak-recall and
   ISRJ AUPR do not drop by more than `0.010` in mean.
5. All five runs preserve exact Q4 non-VFT logits, exact frozen threshold
   vectors, identity separation, and the test-v3 seal.

Failure stops this residual mechanism. Passing permits only a separately
preregistered fresh confirmation source; it does not authorize test-v3.

