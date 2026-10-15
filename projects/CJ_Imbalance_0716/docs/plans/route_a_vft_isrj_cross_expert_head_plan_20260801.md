# VFT-ISRJ cross-expert hard-negative head plan

## Hypothesis

VFT false positives under long-period ISRJ OOD arise because the VFT expert sees
valid time-frequency structure but cannot identify its ISRJ origin.  A tiny
classifier that jointly sees frozen VFT evidence and frozen ISRJ evidence can
learn this cross-label exclusion without changing either feature extractor.

## Model and controls

All heads are five-input logistic regressions with the same six trainable
scalars including intercept.

- aligned: parent VFT logit, VFT expert logit, their delta, ISRJ logit, and
  delta-times-ISRJ-probability interaction;
- capacity: parent/expert/delta plus two generic parent/delta nonlinear terms,
  without ISRJ semantics;
- shuffled: aligned features with ISRJ logits deterministically shuffled within
  every role.

The fit loss weights weak VFT positives by 4 and VFT-negative/ISRJ-positive hard
negatives by 4.  Other samples have weight 1.  `C=0.1`; no search.

Platt calibration uses only the construction calibration role.  Threshold
selection uses only the construction threshold role and maximizes weak recall
subject to overall VFT FPR `<=0.20` and ISRJ-present VFT-negative FPR `<=0.25`.
MFT uses the independently preregistered cap `0.50` with its Q1 parent decision
map.  All other labels use the frozen Q1 decision map.

## Evidence sequence

1. five-seed construction-v3;
2. development-v12 diagnostic replay;
3. ISRJ parameter-OOD diagnostic replay;
4. only if all pass, freeze and generate fresh development-v13.

## Gates on each dataset

- VFT AUPR nonnegative versus Q1;
- VFT weak AUROC gain `>=0.01` and weak recall gain `>=0.01`;
- aligned weak AUROC exceeds both capacity and shuffled controls by `>=0.005`;
- VFT FPR `<=0.20` and ISRJ-present/VFT-negative FPR `<=0.25`;
- unified target weak-recall gain `>=0.03`, target FPR `<=0.20`, and target weak
  AUROC gain `>=0.015` on v3/v12 (`>0` on OOD);
- macro-AUPR gain `>=0.005` on v3/v12 and nonnegative on OOD;
- SawFM probability and all non-target decisions exactly preserved.

Failure stops the cross-expert head; no result-driven threshold changes are
allowed.

