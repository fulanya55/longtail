# H1t stop decision: paired high-to-low soft transfer

Date: 2026-08-03  
Status: **stop H1t on the R1 source; no hyperparameter or protocol retry.**

## Decision rule applied exactly as registered

H1t admission required both a meaningful T1-B0 weak-AUROC gain and a
pairing-specific T1-P1 gain, with paired-bootstrap lower bounds above zero,
at least four positive source splits, no target weak-AUROC decrease larger
than 0.010, and no mean strong-positive AUROC decrease larger than 0.010.

The formal R1 run fails every decisive gain/safety criterion:

- T1-B0 mean weak AUROC is `+0.00076`, not `+0.030`, with CI crossing zero.
- T1-P1 mean weak AUROC is `-0.00137`, not `+0.020`, with only 1/5 positive
  splits and CI crossing zero.
- VFT weak AUROC falls `-0.01178` relative to B0.
- Mean target strong-positive AUROC falls `-0.08791`, far beyond the permitted
  `-0.010` safety bound.

All source-role, composition-preserving permutation, and sealed-test checks
pass. The result is therefore a mechanism failure, not a data-leakage or
execution failure.

## Failure mechanism

The frozen 50 dB teacher has more information, but a detached KL penalty on
weak target positives is not a selective recovery operator. Its gradient flows
through the shared final temporal stage and six-label classifier. It can alter
the common ranking geometry even though the KL mask excludes strong positives
and non-target labels. The observed strong-positive collapse is direct evidence
of this interference.

More importantly, P1 preserves the teacher score distribution, target label,
weakness stratum and full composition while destroying identity matching. T1
does not outperform P1. The small T1-B0 MFT increase is consequently not
attributable to receiver-identity-aligned high-visibility evidence; it is
consistent with a generic score regularization effect. It is neither a
three-target effect nor a safe one.

## Prohibited follow-ups on H1t R1

Do not change or search the KL weight, temperature, mask, composition strata,
power levels, teacher architecture, trainable scope, early-stop key,
calibration, threshold, or permutation on this source. Do not use its result
to select a test-v3 model. No second run on the excluded smoke source is
allowed.

## Research consequence

Generic high-to-low distillation is already established in the literature and
cannot become a project innovation merely by applying it here. This negative
paired-control result removes it as the current route for MFT/VFT weak-evidence
repair. The appropriate next work is either:

1. formulate a **new, non-overlapping physical observability protocol** (for
   example, an aperture/separation regime that changes the evidence available
   for MFT or VFT), then begin again with fresh identity-isolated development
   roles; or
2. narrow the paper to the independently supported RawSaw/SawFM and frozen
   ISRJ aligned-periodicity claims, explicitly excluding a unified six-label
   claim and keeping test-v3 sealed.

Neither option authorizes another capacity, loss, threshold, or generic-KD
variant under the closed current protocol.
