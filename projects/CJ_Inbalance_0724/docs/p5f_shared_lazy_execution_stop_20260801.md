# P5f: shared frozen-forward and lazy-head execution stop report

## Protocol

P5f changed execution scheduling only.  The frozen five P5e base models, thresholds, normalizers, analytic feature bank, and JNR heads were unchanged.  The dense reference executes the base model twice and the head for every row.  The candidate executes one base forward and evaluates the head only for rows with at least one frozen MFT/VFT/ISRJ positive decision.

Three fresh train-only paired-power identity pools (970M, 980M, 990M) were generated and audited before evaluation.  Each has 3,780 identities, 63 combinations, five 10--30 dB views, zero checked overlap with all earlier P4c/P5b--P5e and peer P5f identities, and no test generation/opening/use.  Each P5f comparison uses all five frozen model seeds, CUDA-synchronized timing, five warmups, and fifteen timed repetitions.

## Result

The pre-registered P5f development screen **fails** and is stopped.

| gate family | result |
|---|---|
| Frozen classification hard decisions | exact equality in all 15 data-model pairs |
| Non-target parameter cells | exact equality in all 15 pairs |
| Oracle/predicted-gated NMAE discrepancy | at most `5.96e-8` |
| Shared-lazy / dense median latency ratio | 0.485--0.567; mean 0.541 |
| Union-positive rows evaluated by the lazy head | 0.681--0.727; mean 0.706 |
| Target parameter maximum absolute discrepancy | `5.72e-6` |
| Pre-registered target numerical-equivalence limit | `1.0e-6` |

The failure is numerical, not a decision or metric regression: selecting a union-positive sub-batch changes floating-point matrix-multiplication accumulation relative to the dense head batch.  The largest difference exceeds the frozen exactness tolerance even though it is too small to change a classification decision or a reported NMAE gate.

## Stop decision and evidence boundary

Do not relax the tolerance, change batching, add quantization, modify the head, or rerun the P5f development identities.  Do not generate or use the reserved 1010M/1020M confirmation identities.  P5e remains valid only as dense bounded-JNR attachment evidence; P5f cannot support an output-equivalent lazy-execution or compute-feasibility claim.

The observed latency ratio is diagnostic only, not promotable acceleration evidence, because its paired output-equivalence gate failed.

## Artifacts

- `configs/p5f_shared_execution_lazy_head_preregister_20260801.json`
- `outputs/p5f_shared_execution_lazy_head_20260801/development/result.json`
- `outputs/p5f_shared_execution_lazy_head_20260801/audit_seed970000000.json`
- `outputs/p5f_shared_execution_lazy_head_20260801/audit_seed980000000.json`
- `outputs/p5f_shared_execution_lazy_head_20260801/audit_seed990000000.json`
