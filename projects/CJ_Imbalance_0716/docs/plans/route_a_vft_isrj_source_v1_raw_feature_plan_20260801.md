# Source-v1 fixed raw-feature diagnostic

This run executes the already-frozen `RidgeSTFT` / `RawSource` /
`PhaseScrambledControl` comparison on the fresh source-v1 data after its
identity and estimability audit passed.  It uses exactly the five role
manifests created within each balanced label combination; no development-v3,
OOD-v1, validation, or early result selects a feature parameter.

Fit-only normalization and class-balanced `RidgeClassifier(alpha=10)` are
fixed.  Primary evidence is early hard-view AUROC; validation is descriptive.
Advance conditions are RawSource minus RidgeSTFT >= 0.010 in at least four
role replications, RawSource minus phase-scrambled >= 0.020, and a
validation-to-early gap below 0.150.  Passing only establishes source
identifiability in a fresh controlled diagnostic.  It does not establish
cross-domain generalization or authorize a new unified candidate.
