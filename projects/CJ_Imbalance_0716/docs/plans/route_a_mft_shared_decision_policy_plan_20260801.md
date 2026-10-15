# Route A MFT shared-decision-policy audit plan

## Question

The frozen MFT peak-count residual improved ranking on development-v11, but its
separately fitted Platt calibration and threshold increased high-order false
positives.  Does sharing the H1+B2 MFT calibration and threshold restore the
negative-guard invariant without sacrificing the unified candidate gains?

## Fixed intervention

- Keep the backbone, H1 head, RawSaw branch, ISRJ periodic expert, MFT expert,
  VFT Ridge-STFT expert, and all fusion heads frozen.
- Keep the MFT residual cap at `0.75`.
- Use Q4 calibration and thresholds for every label except MFT.
- Use Q1 H1+B2 calibration and threshold for MFT.
- Perform no fitting, hyperparameter search, calibration, or threshold
  selection in this audit.

This policy has a structural guarantee: because the MFT residual is never
positive and Q1 calibration is monotone, the guarded MFT decision cannot turn a
Q1 negative decision into a positive decision.

## Evidence roles

- development-v3 threshold role: construction audit.
- development-v11: diagnostic reuse after the original frozen candidate failed
  its MFT mechanism gate; it is not independent confirmation after this change.
- development-v12: future identity-isolated confirmation, generated only after
  this audit passes.
- test-v3: sealed and forbidden.

## Preregistered audit gates

The fixed policy must pass on both development-v3 and development-v11:

1. zero MFT negative-to-positive decision flips relative to Q1;
2. non-MFT probabilities and decisions exactly match Q4;
3. mean high-order MFT FPR delta versus Q1 is at most `-0.01`;
4. mean order-5 MFT FPR delta versus Q1 is at most `-0.01`;
5. mean MFT weak-vs-negative AUROC gain versus Q1 is at least `0.01`;
6. mean MFT weak-recall loss versus Q1 is no worse than `-0.02`;
7. unified macro-AUPR gain versus H1 is at least `0.005`;
8. unified target weak-recall gain versus H1 is at least `0.03`;
9. SawFM AUPR and weak recall are unchanged.

If any gate fails, this MFT guard is removed from the promoted candidate.  No
threshold or cap may be tuned against development-v11.

