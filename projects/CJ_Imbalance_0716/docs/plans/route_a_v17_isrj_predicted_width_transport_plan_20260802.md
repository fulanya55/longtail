# Route A v17: frozen predicted-ISRJ-width external transport screen

v16 passed an I/Q width learnability gate on fresh training identities. This next, separate screen tests whether those frozen W-real probes generalize to a new paired physical intervention; it does not alter a classifier.

- Data: a new 1,260-record train-named pair per regime, same seed/labels/nuisance/receiver identity: narrow `10–15 us` versus wide `22–30 us`, replicas fixed to `3–4`.
- Frozen predictors: the five v16 W-real checkpoints and `width_normalizer.json`; no fitting or threshold selection.
- Endpoint: only `VFT=0 AND ISRJ=1` hard negatives. Per frozen probe, record true and predicted widths, paired wide-minus-narrow predicted-width shift, wide-vs-narrow AUROC, R2 and Spearman.
- Gate: mean predicted shift at least `3 us` with positive paired-bootstrap lower CI; mean regime AUROC at least `0.75`, at least 4/5 seeds >= `0.75`; and mean width Spearman >= `0.40`.
- If passed, the next independent experiment may test alignment with frozen Q4 VFT risk. If failed, stop width-conditioned correction. Locked test, test-v2, test-v3 and any v17 test role remain forbidden.
