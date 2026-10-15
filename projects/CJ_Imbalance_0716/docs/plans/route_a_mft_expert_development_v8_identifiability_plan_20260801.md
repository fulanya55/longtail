# Route A MFT expert fresh-v8 identifiability diagnostic

Read-only diagnostic after the M5 candidate failed. Apply the frozen development-v3 strict
final-temporal MFT probe to development-v8 identities without refitting. Report weak-positive
versus all-negative, order>=4-negative, and order-5-negative AUROC for the expert score and U2
score. This diagnostic does not create a candidate or threshold and cannot authorize test-v3.

Interpretation fixed in advance:

- expert hard-negative AUROC >=0.62 with <=0.10 v3-to-v8 drop: fusion/training-support problem;
- expert AUROC <0.62 or drop >0.10: feature evidence does not generalize, so stop MFT head/fusion
  repairs and require a new feature-extraction mechanism.
