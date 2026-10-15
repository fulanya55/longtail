# P5k shared-dense execution freeze

P5k generated, rechunked and audited fresh train-only 4060M, 4090M and 4110M paired-power pools.  Each audit checked 2,482 locally retained historical train files and found zero overlap or unreadable files.  No test data were generated or opened.

The refined guard records non-computing external Python processes but fails only when an external Python PID is also a CUDA-compute PID in `nvidia-smi pmon`.  During the first reference timing loop it observed `p3on_resolution_abstention` (PID 69120) using CUDA and stopped with the planned exception.  This is a genuine concurrent-GPU event, so the elapsed partial work is not a valid cost measurement.

No `result.json` or completed model comparison was produced.  The three P5k development identities are consumed and frozen; P5k confirmation identities 4120M and 4130M must not be generated or used.  The only supported conclusion is that the refined guard blocks a real external CUDA Python workload.  It supplies neither exact-equivalence nor latency evidence.
