# P5g shared-dense execution freeze

P5g was designed as the strict successor to stopped P5f.  It retained the P5e attachment head as a dense, full-batch operation and changed only the duplicated frozen base-model forward into one shared forward.  Its pre-registered gates required exact logits, probabilities, decisions, base parameters, dense-head values, final parameters, and NMAE, plus a CUDA-synchronized end-to-end latency ratio no greater than 0.85.

All three new train-only paired-power pools (3650M, 3660M, and 3670M) were fully generated, rechunked, and audited before inference.  Each audit found 3,780 paired identities, 63 combinations, the expected five-power parameter contract, and zero overlap with each of 2,362 locally retained historical training files.  No test split was generated or opened.

## Freeze event

The one-shot P5g development runner was started but an unrelated P3o-L training process was already sharing the same CUDA device.  Since P5g measures absolute end-to-end inference cost, this violates the intended exclusive-device benchmark condition.  The runner was stopped before it emitted `result.json`; stdout and stderr contain no result or exception.  Therefore there is no P5g equivalence result, latency result, pass/fail decision, confirmation, or test result.

The 3650M/3660M/3670M pools are consumed because frozen-model inference had already begun.  They must not be rerun.  Reserved 3680M/3690M confirmation identities must not be generated or used.  P5e remains narrow dense bounded-JNR attachment evidence; P5f remains the stop result for selected-row lazy execution.

## Artifact

- `configs/p5g_shared_dense_execution_outcome_freeze_20260802.json`
- `outputs/p5g_shared_dense_execution_20260802/development/runner.stdout.log`
- `outputs/p5g_shared_dense_execution_20260802/development/runner.stderr.log`
