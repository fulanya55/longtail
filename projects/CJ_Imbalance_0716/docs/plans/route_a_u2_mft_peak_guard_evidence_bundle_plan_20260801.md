# U2 MFT peak-count guard evidence bundle

Before any independent-test decision, the fresh development-v10 endpoint result is audited along three axes.

1. Statistical replication: compare five-seed development-v3 construction effects with five-seed development-v10 frozen effects. Report direction consistency, bootstrap intervals, standardized effects, exact one-sided Wilcoxon tests, and Holm correction. Treat repeated model seeds across two datasets as replications, not ten independent observations.
2. Subgroup risk: audit MFT recall by relative-power margin, negative FPR by composition order, co-label, and receiver-identity block, with special focus on order 4/5 negatives and margins <=-10 dB.
3. Engineering cost: measure the deployable aligned-reference expert only (one circular FFT matched filter, fixed NMS morphology, one scalar logistic calibrator, and one bounded residual) on GPU and CPU. Wrong-reference and shuffled controls are not deployment components.

No fitting, calibration, threshold selection, or test-v3 access is allowed in this bundle.

