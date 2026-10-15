# Route-A locked-test binding status

Status: `completed_one_time_locked_test_post_read_frozen`.

The prior `binding_required_before_authorization` state in the 20260908 planning document is superseded by this receipt.

- Fresh E-drive source: `generated_data/route_a_locked_test_ama_seed4050000000`
- Powers: 10, 15, 20, 25, 30, 35, 40, 45, 50 dB
- Classes: FMB, AMA, SawFM, SinFM, MFT, ISRJ
- Samples: 2,520 per power, 22,680 total
- Receiver identity intersection with the 18 existing nine-class train/test sources: 0
- Sealed manifest: `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_source_sealed_manifest.json`
- Execution freeze: `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_execution_freeze.json`
- Runner freeze receipt: `outputs/route_a_p1a_anti_alias_Afinal_prelock_20260907/locked_test_runner_freeze_receipt.json`
- Runner: `experiments/run_route_a_locked_test_once.py` (20 prelocked checkpoints, CUDA required)
- Checkpoint integrity: 10 candidate + 10 paired F2_full hashes match the execution freeze
- Model inference on sealed test: completed once under the frozen contract
- `locked_test_used`: true; `read_count=1`

Static gate checks passed: wrong authorization token and wrong execution-freeze hash both fail before any sealed source manifest or MAT path is opened. The runner also refuses a non-empty one-time output root or a repeated terminal receipt.

The one-time read completed after an operational CUDA-device initialization fix. Results are recorded in `outputs/route_a_locked_test_once_20260827/raw_locked_test_result.json` and `docs/reports/route_a_locked_test_result_20260827.md`; terminal receipt status is `completed_one_time_locked_test`. No post-read model, threshold, checkpoint, preprocessing, split, aggregation, or metric changes are permitted.

The only permitted next step is the one-time locked-test runner after explicit user authorization. Model, threshold, checkpoint aggregation, preprocessing, and metric definitions must remain fixed.
