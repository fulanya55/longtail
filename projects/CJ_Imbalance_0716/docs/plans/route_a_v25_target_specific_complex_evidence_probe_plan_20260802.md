# Route A v25: target-specific complex local-evidence probe

## Single hypothesis

After query, residual, fixed-STFT and loss-only routes failed, **H-v25** is that MFT/VFT/ISRJ have target-specific local evidence in raw complex I/Q that is recoverable by target-specific complex filter branches before classifier scores exist. The branch must improve weak-positive-versus-composition-matched-negative ranking over an equal-capacity cyclic-source control while a frozen global path protects the three non-target labels.

## Fresh source and roles

- Generate a new train-only 6-label source with base seed `680000000`, 5,040 unique receiver identities, 80 samples per nonempty label pattern, ISRJ width `[10,30] us`, replicas `[3,4]`, and no test split.
- Use receiver modulo 10 roles: fit `{0..5}` (3,024), early `{6,7}` (1,008), and evaluation `{8,9}` (1,008). Roles are identity disjoint.
- Training uses ordinary six-label presence labels only. Relative component power and weak mask are evaluation/stratification metadata only; neither reaches model input or loss.
- For target `t`, the primary evaluation subset retains weak positives only when their other-five-label combination has both present and absent `t` states in the evaluation role. Negatives are target-absent examples with one of those same other-label combinations.

## Two-stage matched protocol

1. Train one shared TS-CLE global path per seed for 20 epochs with six-label BCE. Select the checkpoint with held-out early six-label BCE.
2. Freeze the complex stem, normalization and global classifier exactly. Train only target local filters/mixers/evidence heads/scales for 20 epochs using the three target presence BCE losses. Candidate routes MFT/VFT/ISRJ local maps to their own labels; control cyclically routes the same three branch maps. Both start from identical local state, use identical minibatches, and select with early target BCE.

Both variants have 5,133 total and the same trainable-local parameter count. FMB/SawFM/SinFM logits must be exact candidate/control equalities. No calibration, threshold, fusion, Q4, old source, locked test, test-v2 or test-v3 is read.

## Promotion gate

Candidate minus cyclic-source control must meet all: mean over MFT/VFT/ISRJ weak AUROC `>=+0.015`, paired bootstrap lower CI >0, >=4/5 positive seed mean deltas; every target mean weak AUROC delta >=0; no target strong AUROC delta below `-0.010` on any seed; non-target logits exact; parameter/trainable contracts, data audit and sealed test-v3 pass. A passing result is expert-level evidence only and permits a separately preregistered six-label integration source. Failure stops TS-CLE without a routing or loss sweep.
