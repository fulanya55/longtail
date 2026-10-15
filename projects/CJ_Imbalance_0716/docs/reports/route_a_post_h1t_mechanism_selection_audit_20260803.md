# Post-H1t mechanism-selection audit: what remains scientifically admissible

Date: 2026-08-03  
Decision: **do not start another network/loss/threshold experiment on the
current single-pulse or closed multi-pulse sources.**

## Why this audit is needed

H1t added a new visibility-paired source and a valid identity-destroying
teacher control. It failed both its target effect and target-specificity gate.
The next action must therefore be selected against all prior negative evidence,
not by cycling to another feature extractor or a different loss.

## Closed mechanism families

| Target | Family | Strongest controlled result | Consequence |
| --- | --- | --- | --- |
| MFT | Single-pulse matched-filter support | H1k weak-MFT candidate-control AUROC `-0.16855`, CI below zero | No ROI, reference, normalization, head or fusion retry |
| MFT | Single-pulse OMP residual | H1m candidate-control AUROC `-0.11215`, CI below zero | No atom/dictionary/regularizer retry |
| MFT | Third-order segmented bicoherence | H0r candidate-control AUROC `-0.09854`, CI below zero while second-order control matches to machine precision | No bin/segment/surrogate/network retry |
| MFT | Four-PRI global coherence-delay readout | H0o absolute AUROC `0.58066` below its pre-registered `0.60` gate | No top-k/window/phase-compensation/learned-head retry |
| VFT | True-timestamp additive projection | H3 true minus permuted weak AUROC `-0.00137`, 0/5 positive | No timestamp/ROI/capacity/loss retry |
| VFT | Full delay-frequency coherence-map concentration | H4 P0 true AUROC `0.51769`, true-minus-permuted `+0.00971`, below both gates | No delay kernel/grid/encoder/threshold retry |
| MFT/VFT/ISRJ | Paired high-to-low posterior transfer | H1t T1-P1 weak AUROC `-0.00137`; strong AUROC `-0.08791` versus B0 | No generic KD, temperature, mask or transfer-weight retry |

All above results are development-only and retain false locked/test-v2/test-v3
use flags. A failed route cannot be revived by changing a design choice that
was an explicit part of its pre-registered control contract.

## Literature boundary

Three primary-paper abstracts/records were checked directly through their
publisher DOI metadata:

- Wang et al. use three consecutive radar pulse periods fused as STFT images
  for CNN-based deceptive-jamming recognition at low JNR. This establishes
  that multi-pulse time-frequency fusion is an existing broad paradigm, not a
  novelty claim for a new VFT backbone. [Sensors 2021](https://doi.org/10.3390/s21082693)
- Wang et al. exploit pulse compression, slow-time Fourier processing, and
  range-Doppler differences for joint range-velocity deception suppression.
  Thus range/velocity joint processing itself is also established. [IET Radar
  Sonar and Navigation](https://doi.org/10.1049/iet-rsn.2018.5158)
- He et al. describe fast-time and slow-time similarity/template processing
  for LFM multiple-false-target jamming. This rules out presenting a generic
  LFM time/frequency-template readout as a new method. [Journal of
  Engineering](https://doi.org/10.1049/joe.2019.0766)

These sources support only the stated scope. They do not validate any project
model or substitute for a controlled project experiment.

## Admissible next research question

The current data contract lacks enough attributable weak MFT/VFT evidence for
another learning modification. A future experiment must change the **physical
information source**, not its network readout:

1. **New sensing contract.** Obtain or simulate a receiver setting with an
   additional independently modeled observation channel (for example,
   multi-receiver spatial/geometry diversity) or an experimentally justified
   coherent aperture. The exact physical variable, its generator, and its
   relation to MFT/VFT must be defined before any labels, architecture or
   training are inspected.
2. **Fresh H0 only.** On fresh receiver identities, construct paired
   present/absent records with all non-target components, noise and generator
   streams fixed. Compare a fixed physical statistic to a control that erases
   only the proposed physical relation. Require absolute discrimination and
   candidate-control discrimination before an H1 network exists.
3. **No novelty shortcut.** A subsequent H1 must compare the physical input to
   an equal-parameter information-destroying control, preserve the other five
   labels through a safe route, and use five identity-grouped development
   seeds. Generic multi-pulse fusion, STFT/CNN, range-Doppler processing or KD
   cannot be named the innovation.
4. **Only then consider unification.** An admitted H1 is still a component
   result. It must be followed by fair C0--C4 six-label baselines, causal
   ablations, robustness evidence, frozen calibration/thresholds, and only
   then one test-v3 authorization.

## Current paper-safe scope

Without a new sensing contract, the defensible direction is a narrower paper:
RawSaw as a label-local SawFM route plus the frozen aligned-periodicity ISRJ
component under its stated stress conditions. It must report the MFT/VFT/H1t
closures and may not claim a validated unified six-label VTSE-VBF method or
use test-v3.

## Required authority before the next experiment

The project does not currently contain multi-receiver/spatial measurements or
an approved generator contract for a new sensing channel. Creating such a
dataset is a material scope expansion beyond model implementation. It requires
an explicit choice of the physical sensing change and permission to generate
or acquire the corresponding fresh data. Until then, the correct action is
documentation/literature work, not another model run.
