# 2025--2026 literature refresh and compatibility audit

Date: 2026-08-03  
Purpose: update the novelty boundary before any new VTSE-VBF experiment. This
is a literature/compatibility audit only; it reads no I/Q, labels, predictions
or sealed test data.

## Evidence tiers

`[F]` means the original publisher DOI metadata and abstract were retrieved.
`[T]` means only publisher/DOI title metadata was available; no technical
detail is inferred from that title. These papers do not establish performance
for the present project.

| Work | Verified scope | Relation to Route A | Tier |
| --- | --- | --- | --- |
| [PG-VCIN, Sensors 2026](https://doi.org/10.3390/s26061900) | Its abstract describes physics/statistics plus time-frequency representations, causal-confounder treatment, variational bottleneck and dual latent intervention constraints for **few-shot** radar-jamming recognition on a semi-physical simulation dataset. | “Physics-guided”, “causal intervention”, or generic invariance cannot be claimed as the novelty of a future Route-A model. It neither reports the present six-label visibility-tail protocol nor supplies a compatible baseline. | [F] |
| [Multi-feature lightweight compound-jamming recognition, Sensors 2026](https://doi.org/10.3390/s26041296) | Its abstract describes three time-frequency representations, parallel multi-scale branches, attention and weighted fusion for seven simulated compound-jamming types at low JNR. | Generic multi-representation, multi-branch, attention or lightweight fusion is established. It does not prove label-internal weak/strong stratification, identity isolation, or safe six-label residual fusion. | [F] |
| [Category-Specific Selective Feature Enhancement, ICCV 2025](https://doi.org/10.1109/ICCV51701.2025.00358) | Publisher title/record identifies a category-specific feature method for long-tailed multi-label image classification. | Label-specific feature enhancement is a close conceptual neighbour. Until the full method and code are inspected, it is not a fair numeric baseline and must not be reimplemented under its name. | [T] |
| [Channel-hierarchical GCN with semantic alignment, Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.132832) | Publisher title/record identifies a graph/semantic-alignment long-tailed multi-label image method. | The semantic graph/image representation is not available in the raw complex-I/Q F2 interface. A partial port would be a new custom method, not this baseline. | [T] |
| [Prompt distributions and probability bias, Knowledge-Based Systems 2026](https://doi.org/10.1016/j.knosys.2026.115363) | Publisher title/record identifies prompt distributions and probability bias for long-tailed multi-label image recognition. | Prompt/vision-language assumptions are unavailable in this waveform-only task; it is literature context, not a same-protocol baseline. | [T] |
| [DP-VLPA, AAAI 2026](https://doi.org/10.1609/aaai.v40i8.37600) | Its abstract describes visual-language pretraining, tail-aware text generation, dynamic query reweighting and co-occurrence-aware loss on VOC-LT/COCO-LT. | It confirms current LTML work handles tail-class and co-occurrence bias using language/visual supervision. It does not address physical visibility strata and cannot be fairly run without adding unavailable image/text inputs. | [F] |

## Consequences for the proposed contribution

The following statements are prohibited in a Route-A paper:

1. first physics-guided or causal radar-jamming recognition;
2. first multi-scale/multi-branch/attention fusion for compound jamming;
3. first label-specific feature enhancement for long-tailed multi-label data;
4. a claimed fair numerical comparison to a vision-language/graph/prompt
   method after only transplanting a loss or a head.

The only still distinguishable scientific hypothesis remains narrower:

> When a target label has a physically defined **within-label visibility
> distribution**, a target-local temporal statistic plus a three-stratum
> weak-positive/strong-positive/negative objective can improve weak evidence
> without inference-time visibility metadata, without a capacity explanation,
> and without damaging the other five label decisions.

This is a hypothesis, not a current contribution claim. Its evidence would
require a promoted unified C3 candidate that beats C0/C4 on the pre-registered
development-v3 gates. Current C3/C4 and H1t results do not meet that standard.

## Compatibility decision

The official 0716 generator and the available F2 interface are single-receiver
complex I/Q. They do not contain visual-language prompts, semantic graphs,
multi-receiver spatial arrays, or a second independently calibrated sensing
channel. Adding any of those after seeing Route-A failures would be a new data
and sensing protocol, not a harmless architecture adjustment.

Therefore this refresh authorizes neither a new network nor a test-v3 run. It
does require that any future paper separate:

- existing generic physical/causal/multi-feature ideas;
- the unverified label-internal visibility-tail hypothesis; and
- the currently verified narrow RawSaw and ISRJ component evidence.

## Retrieval record

The source metadata and abstract records were retrieved on 2026-08-03 from
the corresponding DOI metadata services. No unavailable full text was
paraphrased as verified method detail.
