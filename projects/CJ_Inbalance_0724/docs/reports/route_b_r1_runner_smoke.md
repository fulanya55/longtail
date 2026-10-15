# Route B R1 trainer/aggregator runner smoke

- Status: **verified_runner_smoke_only**
- Run: `outputs/route_b_r1_runner_smoke/r1_20260826_121902`
- Variants exercised: B0 and B4
- Seeds exercised: 2041--2045
- Epochs: 1
- Samples per combination: train 2 / validation 1
- Device: CPU
- Locked test opened: `false`

The smoke confirms dataset construction, seed looping, classification-only and
warmup joint paths, checkpoint writing, per-seed CSV writing, and incomplete-run
aggregation. These values are not performance evidence. Formal R1 remains
pending for B0--B4, five seeds, the frozen development split, and the declared
hard gates.
