# Route A RawSaw top-k 4/8/16 within-run protocol repair result

The five-seed equal-budget top-k experiment is valid as a pooling sensitivity
ablation: all arms have equal trainable parameter count, identical identity
roles/calibration/threshold protocol, and exactly identical probabilities and
predictions for the five non-SawFM labels. No locked test, test-v2, or test-v3
was used.

The historical top-k=16 checkpoint is not bitwise reproducible after CUDA
training (per-seed maximum state differences `0.04577--0.13899`). This remains
an explicit numerical-reproducibility limitation, but is not a reason to
discard the simultaneously trained 4/8/16 contrast.

Against within-run top-k=16, top-k=4 has SawFM weak-AUROC `-0.02024`
(`[-0.04225,+0.00274]`) and top-k=8 has `-0.00798`
(`[-0.03224,+0.01142]`). Neither interval selects a robust winner; top-k=16
has the strongest point estimate but cannot be promoted or changed from this
development-only sensitivity experiment.

Artifacts: `docs/plans/route_a_rawsaw_topk_within_run_protocol_repair_plan_20260801.md`,
`experiments/audit_route_a_rawsaw_topk_within_run_protocol_repair.py`, and
`outputs/tapc/route_a_rawsaw_topk_within_run_protocol_repair/route_a_rawsaw_topk_within_run_protocol_repair_summary.json`.
