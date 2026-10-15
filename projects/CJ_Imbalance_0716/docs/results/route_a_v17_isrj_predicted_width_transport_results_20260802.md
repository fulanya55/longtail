# Route A v17: frozen predicted-ISRJ-width external transport

## Decision

**Pass.** The v16 width predictors, frozen without refitting, transport to new paired narrow/wide ISRJ data. This validates a predicted I/Q width signal outside its development identities. It still is not a classifier change and does not establish that width correction improves VFT recognition.

## Protocol boundary

- Preregistration: [`route_a_v17_isrj_predicted_width_transport_plan_20260802.md`](../plans/route_a_v17_isrj_predicted_width_transport_plan_20260802.md).
- New paired source: `复合干扰数据0802_v17_isrj_width_transport`, 1,260 train-named records per regime, same receiver identities `[728677743,728679002]`, labels and ISRJ-absent I/Q exactly matched.
- Narrow: `10–15 us`; wide: `22–30 us`; replicas fixed to `3–4`.
- End point: the 320 `VFT=0 AND ISRJ=1` hard negatives per regime.
- Inputs are evaluated by the five frozen v16 W-real checkpoints plus the frozen fit-only normalizer. No model, calibration, threshold or fusion parameter is fit.
- All receipts record `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`, and `test_v17_used=false`.

## Results

| Metric | Mean across frozen v16 seeds | 95% bootstrap CI |
|---|---:|---:|
| Wide minus narrow predicted width | +8.469 us | [+8.229, +8.701] us |
| Wide-vs-narrow regime AUROC | 0.9639 | [0.9611, 0.9671] |
| Combined true-width R² | 0.6375 | [0.6233, 0.6535] |
| Combined width Spearman | 0.7595 | [0.7518, 0.7684] |

Every frozen predictor has regime AUROC >=0.75 (5/5). All pre-registered transport gates pass.

## Meaning and next gate

Together, v16 and v17 establish two separate facts: sample width is learnable from I/Q on held-out identities, and that predicted evidence transfers to a new paired physical width intervention. The still-unanswered question is decision relevance: does the predicted width increase exactly where the **frozen** Q4 VFT hard-negative score/FPR increases?

The next screen must evaluate the existing frozen Q4 policy on this same new paired data and report aligned paired width-score and VFT-risk effects. It must not fit a residual, tune a threshold, or access test-v3. Only an alignment pass can justify designing a new, fresh-data bounded residual correction.

## Artifacts

- Paired identity audit: `outputs/tapc/route_a_v17_isrj_width_transport_generation/identity_parameter_pair_audit.json`.
- Frozen predictor summary: `outputs/tapc/route_a_v17_isrj_predicted_width_transport/route_a_v17_isrj_predicted_width_transport_summary.json`.
- Per-probe predictions: `outputs/tapc/route_a_v17_isrj_predicted_width_transport/seed_*/predictions.npz`.
