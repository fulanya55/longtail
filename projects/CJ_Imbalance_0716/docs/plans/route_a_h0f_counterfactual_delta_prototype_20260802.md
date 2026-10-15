# Route A H0f: Counterfactual Delta Prototype (CDP) representation gate

## Why this is not H0e again

H0e stopped a direct pair-to-logit margin: it only required
`z_positive,t > z_negative,t` and therefore let the pair relation collapse
into the same scalar decision boundary. H0f changes the supervised object and
the inference object. It learns a **temporal feature-difference prototype**
from `h(x_positive)-h(x_negative)` during training, then uses the learned
prototype to read a single-view temporal feature at inference. No pair id,
power, weak mask, physical parameter, second view, classifier query, fixed
STFT image, or target logit is used to construct this prototype evidence map.

This is a new representation-level counterfactual hypothesis, not a retry of
the stopped H0e hyperparameters. H0e data, code outputs, targets and identity
source are not reused.

## Single falsifiable hypothesis

For a nuisance-matched pair that differs only by MFT, VFT or ISRJ, the pooled
temporal feature difference has a target-specific direction. If that direction
is explicitly bound to a learnable prototype, the prototype projection of a
single I/Q view should improve weak-positive versus matched-negative ranking
over both ordinary BCE and a target-binding-permuted control.

## Fresh source and roles

- New base seed `750000000`, 2,400 pair identities / 4,800 views; it has the
  same *predeclared* generation contract as H0e but a fully disjoint identity
  range. It is a new source, not a resplit or reused source.
- For each MFT, VFT and ISRJ target, 800 pairs use five fixed nuisance
  combinations. Pair members share component substreams, reference and
  receiver noise; positive view only adds its target.
- Common JNR is exactly 20 dB and target offset is exactly -12 dB; these are
  generation/audit metadata, not model inputs. Fit/early/evaluation use
  receiver modulo 10 at 1,440/480/480 pairs; a pair cannot cross roles.
- No test data are generated or read. Audit must prove zero receiver identity
  overlap with every historical train source and an untouched test-v3 seal.

## Fixed model and controls

`CounterfactualDeltaPrototypeProbe` has a raw-I/Q temporal encoder, six-logit
global head, 16-D temporal projector, and three 16-D learnable prototypes.
The target output is global target logit plus a bounded prototype projection;
FMB/SawFM/SinFM are global-safety outputs. It has the same architecture,
parameters, initialization, samples, batches and optimizer in every arm.

For pair target `t`, with `e(x)` the projected pooled temporal feature and
`p_t` the normalized prototype:

`d = normalize(e(x_positive)-e(x_negative))`

`L_delta = CE(d @ normalize(P)^T / 0.10, binding(t))`.

All arms use six-label BCE. `B0` adds no delta loss; `B1_binding_permuted`
uses a fixed cyclic target binding `{MFT->VFT,VFT->ISRJ,ISRJ->MFT}`; `C_CDP`
uses the true binding. The delta CE weight is fixed at `0.20`; this is neither
a pair-ranking loss nor a direct logit margin. AdamW lr `3e-4`, weight decay
`1e-4`, 20 epochs, 48 pairs/batch, five seeds 2039--2043, early target BCE.
There is no threshold, calibration, fusion or test evaluation.

## Gate and stop rule

The candidate must beat **both** B0 and B1 on mean MFT/VFT/ISRJ weak AUROC by
`+0.015`, have a positive paired bootstrap 95% lower CI and at least 4/5
positive seeds. Every target mean weak-AUROC delta must be non-negative, every
seed all-positive AUROC delta must be at least `-0.010`, and candidate heldout
true-binding delta accuracy must be `>=0.60`. Parameter/init/role/test-seal
contracts must match.

Failure stops CDP and forbids changing its source, projector width, prototype
temperature, delta-loss weight, nuisance combinations, target offset or fusion.
Success permits only a separately generated fresh six-label integration study.
