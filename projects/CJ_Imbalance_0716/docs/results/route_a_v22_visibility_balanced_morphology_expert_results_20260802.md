# Route A v22: visibility-balanced morphology expert result

## Decision

**Stop weak/strong positive-mass reweighting for the morphology expert.** The same-capacity VBF-trained expert reduces, rather than improves, held-out weak-VFT-versus-hard-negative ranking. It must not be attached to Q4.

## Protocol and boundary

v22 used a newly generated train-only source (base seed `620000000`; SHA-256 `055b6b6c3f7f614efa17e6c196f949d7f71ce1492a58803a1378ad04a5542627`) with 5,040 unique receiver identities. The ISRJ-present fit/early-stop/evaluation roles were 1,536/512/512 identities. Evaluation contained 66 weak VFT positives, 190 strong VFT positives, and 256 VFT-absent/ISRJ-present hard negatives.

`M_BCE` and `M_VBF` used exactly the same frozen 268-dimensional STFT feature and `268 -> 64 -> 1` GELU head (17,281 parameters), initialization, optimizer, learning rate, batch size, epochs, and three-stratum early-stop criterion. The sole change was the fit-loss weight: VBF assigned weak VFT positives the recorded deterministic weight `579 / 189 = 3.06349`, equalizing weak and strong positive total loss mass. No Q4 output, residual, calibration, threshold, v19-v21 data, historical locked test, test-v2, or test-v3 was read.

## Five-seed held-out result

| Metric: M_VBF minus M_BCE | Mean delta | 95% paired bootstrap CI | Result |
| --- | ---: | ---: | --- |
| weak VFT versus hard-negative AUROC | `-0.00778` | `[-0.01351,-0.00199]` | significantly adverse |
| weak VFT versus hard-negative AUPR | `-0.00765` | recorded in summary | adverse |
| strong VFT versus hard-negative AUROC | `-0.00604` | recorded in summary | within the per-seed safety bound |
| overall VFT AUROC | `-0.00649` | recorded in summary | one or more seed safety violations |

Only one seed showed a positive weak-AUROC delta (`+0.00207`); the other four were negative. The candidate therefore failed the weak-AUROC magnitude, confidence-interval, 4-of-5 consistency, weak-AUPR, and overall-safety gates. Architecture matching and test-v3 sealing passed. The preregistered decision is `stop_visibility_balanced_morphology_training`.

## Mechanism conclusion

Equalizing weak and strong positive *mass* is not equivalent to separating weak positives from the matched VFT-absent/ISRJ-present negatives. The weighting pushes all weak-positive probabilities together but does not directly optimize their ordering against the error-causing negative stratum; it also slightly degrades global ordering. Thus neither a new class weight nor a new scalar fusion coefficient is justified.

The next distinct hypothesis is a weak-positive/hard-negative pairwise ranking term added to ordinary BCE while retaining strong-positive BCE. It will be tested on another newly generated identity source with the same feature/head capacity and a fixed, preregistered pair-loss coefficient. It is not a threshold or weight sweep.

## Artifacts

- Plan: `docs/plans/route_a_v22_visibility_balanced_morphology_expert_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v22_visibility_balanced_morphology_expert.m`, `experiments/audit_route_a_v22_visibility_balanced_morphology_expert.py`, `experiments/run_route_a_v22_visibility_balanced_morphology_expert.py`
- Formal artifacts: `outputs/tapc/route_a_v22_visibility_balanced_morphology_expert/`
