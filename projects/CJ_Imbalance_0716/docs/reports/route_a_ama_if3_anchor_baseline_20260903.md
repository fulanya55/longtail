# Route-A AMA 六类 IF3 与 Anchor 开发基线

## Protocol

- Data: `six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA`.
- Classes/order: `FMB, AMA, SawFM, SinFM, MFT, ISRJ`.
- Input: true complex I/Q `[B,2,3000]`, per-sample RMS normalization.
- Backbone: `F2_DFMS_Full` (`dfms_iq`, width=16, feature_dim=128, dropout=0.2, 257,511 parameters).
- Initialization: frozen F2 backbone from the Route-A seed-2039 checkpoint; six-class classifier reinitialized; then full-model fine tuning.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: AdamW, `lr=1e-4`, `weight_decay=1e-4`, batch=32, 8 epochs, gradient clipping=5.
- Split: deterministic combination-stratified 80/20 split of the 8,000 training samples (6,415 fit / 1,585 validation); all 63 combinations retained.
- Anchor: uniform-combination `WeightedRandomSampler` with replacement over the same fit indices, same sample budget and seeds. This is a sampling control, not an independently generated balanced raw dataset.
- Development test: all 8,000 samples in `closedset_6class_test.mat`; no locked test opened.

## Results (three seeds)

| condition | val Macro-AUPR | val Macro-F1 | val combo exact | val Hamming | test Macro-AUPR | test Macro-F1 | test combo exact | test Hamming |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IF3 ordinary shuffle | 0.975414 ± 0.000988 | 0.916482 ± 0.004475 | 0.610726 ± 0.016941 | 0.080547 ± 0.004477 | 0.975502 ± 0.000860 | 0.916220 ± 0.004495 | 0.613708 ± 0.016088 | 0.081257 ± 0.004497 |
| Anchor uniform-combination sampler | 0.972337 ± 0.000806 | 0.910153 ± 0.004411 | 0.586540 ± 0.027791 | 0.085980 ± 0.007139 | 0.972754 ± 0.000661 | 0.909399 ± 0.004749 | 0.589375 ± 0.028001 | 0.086743 ± 0.007307 |

Paired test differences (IF3 minus Anchor, seeds 2039/2040/2041):

- Macro-AUPR: `+0.002747`
- Macro-F1: `+0.006822`
- Combination exact accuracy: `+0.024333`
- Hamming loss: `-0.005486`

## Interpretation boundary

The IF3 ordinary-shuffle run is slightly better than the Anchor sampler on this development split. This does **not** prove that label imbalance improves performance or that IF3 has no adverse effect. The Anchor condition oversamples rare combinations with replacement, whereas IF3 uses each fit sample once per epoch; the comparison therefore includes a sampler/duplicate-exposure effect. The current result is a development screening result only. A pure label-causal comparison requires independently generated or constrained manifests with matched combination coverage and a preregistered balanced control.

The data contract itself passed: IF=3.0, CF=7.8537, all 63 combinations, train/test receiver-seed intersection=0, and exact signal-hash intersection=0.

## Receipts

- Summary: `outputs/route_a_ama_if3_anchor_baseline_20260903/summary.json`
- Terminal receipt: `outputs/route_a_ama_if3_anchor_baseline_20260903/terminal_receipt.json`
- Training script: `experiments/run_route_a_ama_if3_anchor_baseline_20260903.py`
- Locked test used: `false`
- Frozen E5 modified: `false`
