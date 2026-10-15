# Asymmetric visibility transfer: literature and novelty admission audit

## Scope and source tiers

This audit asks whether H0t permits a novel-method claim. It does not permit
claiming knowledge distillation itself as new. `[F]` denotes a directly read
original source; `[T]` denotes bibliographic metadata only and must not support
a method-detail claim.

| Source | Verified content | Consequence for H1t | Tier |
| --- | --- | --- | --- |
| Hinton et al., *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531 | The original paper introduces a teacher's soft targets as a training signal. | A high-visibility teacher and low-visibility student are standard KD, not an innovation. | [F] |
| Lopez-Paz et al., *Unifying distillation and privileged information*, arXiv:1511.03643 | The original abstract frames generalized distillation as learning from multiple machines/data representations, including privileged information. | A train-only high-JNR view is a privileged representation; its use must not be presented as a new paradigm. | [F] |
| Zhang et al., *Few-Shot Radar Jamming Recognition Network via Time-Frequency Self-Attention and Global Knowledge Distillation*, IEEE TGRS, DOI 10.1109/TGRS.2023.3280322 | Title, venue and DOI were resolved to the IEEE record. Full method text was not obtained in this audit. | Radar-jamming KD is a close topical precedent. Do not claim generic radar-jamming KD novelty; retrieve full text before any paper novelty statement. | [T] |
| *ClST: A Convolutional Transformer Framework for Automatic Modulation Recognition by Knowledge Distillation*, IEEE TWC, DOI 10.1109/TWC.2023.3347537 | Title and DOI metadata only. | Confirms KD is active in I/Q recognition; no architecture or loss detail is asserted here. | [T] |

## Admission decision

The following claims are prohibited:

1. “high-SNR teacher to low-SNR student” is a new method;
2. “knowledge distillation for radar jamming” is new;
3. the H0t score gain proves a learned low-visibility improvement.

The only potentially publishable, falsifiable contribution is narrower:
**under an identity-paired compound-jamming protocol, test whether observed
label-internal visibility imbalance can be reduced by target-local,
positive-supported, asymmetric posterior transfer, beyond a same-capacity
teacher-pair permutation control, without using power metadata at inference.**

That statement remains `[T]` until H1t passes. It needs direct evidence that
the paired, label-local teacher signal (not capacity, marginal teacher scores,
or a relaxed threshold) improves weak-target ranking on a fresh source.

## H1t compulsory controls

- B0: identical student initialization and supervised Focal objective only;
- T1: B0 plus stop-gradient, target-local binary posterior KL on true target
  positives, using the matched 50 dB view only during training;
- P1: T1 with the high-view teacher posterior cyclically permuted **within the
  same positive target and composition stratum**, preserving model capacity and
  score marginals but destroying identity pairing;
- all arms use the same fresh identity roles, epochs, optimizer and rankings;
  no calibration or threshold is permitted for the primary H1 endpoint.

H1t may not reuse development-v3 H0t logits for training or selection. It
must first generate a new identity-isolated train-only three-level source.
test-v3 remains sealed.

## Source links

- [Hinton et al. original paper](https://arxiv.org/abs/1503.02531)
- [Lopez-Paz et al. original paper](https://arxiv.org/abs/1511.03643)
- [TGRS radar-jamming record](https://doi.org/10.1109/TGRS.2023.3280322)
- [TWC modulation-recognition record](https://doi.org/10.1109/TWC.2023.3347537)
