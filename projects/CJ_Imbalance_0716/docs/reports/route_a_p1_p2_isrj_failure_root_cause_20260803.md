# Route A: P1/P2 ISRJ H0 failure root-cause audit

Date: 2026-08-03  
Nature: read-only analysis of the already closed P1 and P2 sources. It did not
create a candidate, select a parameter, fit a model, read a test set, or alter
either protocol.

## Main finding

The P1/P2 ISRJ failure is not explained by a missing generic temporal pattern.
Both true-order scores are far above their respective destruction controls for
**absent**, weak and strong views. The failed operation is label attribution:
in compound backgrounds, the absent view already contains a large periodic or
correlation score, so adding an 8 dB ISRJ component has near-zero, and often
negative, paired increment. A 20 dB ISRJ component is detectable, which shows
a visibility/mixture limit rather than an I/Q reader failure.

## Arm-level evidence

| Protocol | absent mean score | weak mean score | strong mean score | weak - absent | strong - absent |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 generic envelope ACF | 0.3276 | 0.3516 | 0.6030 | +0.0240 | +0.2754 |
| P2 high-replica joint periodicity | 0.2183 | 0.2442 | 0.4393 | +0.0258 | +0.2210 |

P2's fixed block-shuffle control is low for every arm (mean approximately
0.029--0.031). Thus block shuffling proves the statistic uses temporal order,
but it does not prove that the temporal pattern belongs to the ISRJ label.

## Compound-background stratification

The key diagnosis is the paired weak-minus-absent increment by the number of
nuisance labels in the matched absent view:

| Nuisance label count | P1 weak delta, mean | P2 weak delta, mean |
| --- | ---: | ---: |
| 1 | +0.0913 | +0.1038 |
| 2 | -0.0005 | +0.0093 |
| 3 | -0.0068 | +0.0004 |
| 4 | +0.0000 | -0.0034 |
| 5 | +0.0029 | -0.0019 |

Strong ISRJ remains positive at these same composition orders, whereas weak
ISRJ loses its increment after two coexisting labels. This directly explains
the failed weak AUROC and paired-direction gates: a scalar periodicity score
cannot separate weak ISRJ evidence from periodic/correlated background
structure, even when the physical ISRJ condition is made more favorable.

## What this rules out

1. Do not describe P2's high-replica condition as a shared all-target
   observability success; only its MFT/VFT H0 rows pass.
2. Do not retune P1/P2's lag range, score combination, block size, permutation,
   slice width, replica count, JNR, PRI or a learned head. Those would be
   post-result searches on already consumed P1/P2 sources.
3. Do not convert the existing H0s frozen B2 result into a claim that raw
   periodicity is a generally transportable weak-ISRJ observable. H0s supports
   a narrow trained/frozen aligned-periodicity component only.

## Consequence for the full objective

At present, the required unified claim has no admissible all-target physical
entry point:

- original 0716 single-pulse MFT/VFT has inadequate aperture;
- H0v, P1 and P2 do not provide a shared weak ISRJ attribution statistic;
- P1/P2 are closed and cannot yield C0/C3/C4 training; and
- the sealed original test-v3 cannot be used to resolve this.

Meaningful continuation needs an external state change, not another local
hyperparameter or statistic variant: an independently specified sensing/data
protocol with label-attributable weak ISRJ evidence (for example, an additional
calibrated channel, a controlled receiver setting, or real/semi-physical
measurements), or explicit authorization to change the paper objective to the
already supported narrow RawSaw + frozen ISRJ component evidence.

## Artifacts

- Diagnosis script: `experiments/analyze_route_a_p1_p2_isrj_h0_failure.py`
- Machine receipt: `outputs/tapc/route_a_p1_p2_isrj_h0_failure_diagnosis/summary.json`
- P1 result: `docs/results/route_a_p1_observability_adequate_h0_results_20260803.md`
- P2 result: `docs/results/route_a_p2_full_target_h0_results_20260803.md`

The diagnostic unit tests passed (`6 passed`). All receipt test-use flags are
false.
