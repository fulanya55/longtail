# Route A U2 development-v7 frozen stratified robustness preregistration

## Scope

Analyze only the already saved frozen development-v7 predictions. No model inference,
training, calibration, threshold selection, seed selection, or test access is allowed.

## Axes

For MFT, VFT, and ISRJ separately:

- relative component-power gap: <=-15, (-15,-10], (-10,-5], (-5,0], >0 dB;
- JNR: [10,15), [15,20), [20,25), [25,30.1) dB;
- physical parameter: MFT target count 3-4/5-6/7-8, VFT Doppler 1-3/3-5/5-7 kHz,
  ISRJ sample width 5-10/10-15/15-20.1 us;
- composition order 1 through 6;
- each co-label present and absent;
- receiver-noise-seed modulo-five cohort.

For positive-only bins, all negatives remain in the comparison pool. A group is eligible
for ranking metrics only with at least 40 positives, 100 negatives, and 20 weak positives.

## Frozen comparisons

- U2 versus U0: complete unified method versus strongest direct baseline.
- U2 versus U1: incremental aligned-periodicity contribution.
- U2 versus U3: aligned-periodicity versus block-shuffled control.

## Gate fixed before subgroup metrics are read

1. U2-vs-U0 weak-AUROC is positive in at least 70% of supported target/group cells.
2. U2-vs-U0 weak recall is nonnegative in at least 70% of supported target/group cells.
3. Overall weak-AUROC gain is positive for each of MFT, VFT, and ISRJ.
4. Mean weak-recall gain in the two weakest relative-power bins is at least 0.020.
5. At least 9/12 supported target/JNR cells have positive weak-AUROC gain.
6. At least 12/15 target/receiver-cohort cells have positive weak-AUROC gain.
7. For ISRJ, U2-vs-U1 and U2-vs-U3 weak-AUROC is positive in at least 70% of supported cells.
8. Worst supported composition-order FPR is <=0.30 and its U2-vs-U0 increase is <=0.05.
9. In both SawFM-present and SawFM-absent subsets, target-label mean AUPR decline is no worse than 0.005.
10. SawFM logits, probabilities, and predictions remain exactly invariant.
11. All inputs declare external_refit=false and test-v3 remains sealed and unused.

Failure is reported without subgroup-specific repair on development-v7.
