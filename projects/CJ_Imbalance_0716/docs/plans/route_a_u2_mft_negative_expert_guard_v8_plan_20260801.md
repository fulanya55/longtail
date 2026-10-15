# Route A U2 MFT negative-evidence expert guard preregistration

## Hypothesis

The previously validated final-temporal MFT expert contains complementary physical evidence, but
its signed residual can raise both weak positives and hard negatives. On top of U2, use only its
negative evidence:

`residual = 0.5 * sqrt(4p(1-p)) * min(tanh(z_mft), 0)`

and add the residual only to the MFT logit. Positive expert evidence cannot raise a score. The
frozen U2 Platt calibration and threshold are reused without refitting in the construction guard.

## Controls

- G0: frozen U2.
- G1: aligned MFT negative-evidence guard; fixed candidate.
- G2: role-wise shuffled expert z with the same formula.
- G3: aligned signed expert residual, showing whether one-sidedness is the safety mechanism.

No trainable parameters are added. All non-MFT logits are exactly invariant.

## Fail-fast guard before fresh development-v8

On the existing v3 early role, G1 must satisfy:

- macro-AUPR decline <=0.005;
- target mean weak-recall decline <=0.010;
- target mean weak-AUROC decline <=0.005;
- MFT order>=4 FPR decreases by at least 0.020;
- non-MFT logits are exact;
- aligned G1 weak-vs-hard-negative AUROC is no worse than shuffled G2.

This is only a construction check. If it passes, evaluate the frozen G0-G3 once on fresh
development-v8 and apply the original unified and worst-order FPR gates. test-v3 remains sealed.
