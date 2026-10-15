# Frozen U2 MFT peak-count guard confirmation on development-v10

The complete U2 endpoint, five fit-only one-dimensional peak-count calibrators, residual cap 0.75, U2 calibration, and U2 thresholds are frozen before generating development-v10. Only the train split with base seed 104,000,000 is generated. Expected receiver-noise identities are 302,677,743--302,702,942 and must be disjoint from all prior roles. No test-v10 split is generated or opened; test-v3 stays sealed.

For each seed 2039--2043, U2 is reconstructed from frozen checkpoints and experts. The aligned peak-count guard, a wrong-chirp same-capacity control, and a deterministic shuffled aligned-evidence control are applied without refitting. All non-MFT logits must be exact.

The preregistered confirmation gate requires: macro AUPR delta >=-0.002; target mean weak-recall delta >=-0.010; target mean weak-AUROC delta >=0.005; MFT order>=4 FPR delta <=-0.020; MFT order-5 FPR delta <=-0.020; hard AUROC delta >=0.010; aligned-minus-wrong and aligned-minus-shuffled hard AUROC each >=0.005; at least four seeds reduce hard FPR; and exact non-MFT preservation.

Passing establishes fresh-identity endpoint improvement on development evidence. It still does not authorize test-v3. A final independent-test decision requires the complete paper evidence audit and a predeclared one-shot rule.

