# P3o-K MATLAB generator repair

## Root cause

The MATLAB crash was secondary. The preserved HDF5 log identifies the first fault as `errno=28: No space left on device` while saving `closedset_6class_train.partial.mat` on C:. MATLAB R2024b then faulted during HDF5 shutdown (`0xc0000005`). This was not a waveform-generator, identity, or HDF5 concurrency defect.

## Repair and verification

- Archived the four incomplete frozen P3o-J/P3o-K generation directories to `D:\CJ0724_frozen_generation_artifacts\20260802`; no artifacts were deleted.
- Restored the accidentally truncated MATLAB wrapper and replaced its corrupted generator-root literal with the verified source root.
- Added the optional `outputBase` argument. Future runs must write to `D:\CJ0724_generated_data`, where free capacity exists, instead of C:.
- Ran `executeGeneration=false` with an explicit D: root for five regimes and three powers. All 15 manifests were constructed successfully; no sample file was generated and no identity was consumed.

## Preventive I/O guard (2026-08-03)

- The generator now defaults to `D:\\CJ0724_generated_data` and still accepts an explicit `outputBase` for an approved run.
- Before any manifest or HDF5 transaction, it verifies the generator source path, creates the output directory if needed, and performs a create/close/delete write probe in that directory.
- For `executeGeneration=true`, it also requires at least 15 GiB free on the selected output volume. A capacity failure therefore raises a MATLAB error before a partial v7.3 file is opened, instead of surfacing later as an HDF5 shutdown access violation.
- MATLAB R2024b preflight was rerun with the repaired wrapper and D: output (`executeGeneration=false`, five regimes, three powers). It printed `P3OK_IO_GUARD_PREFLIGHT_PASS`; no waveform sample, test data, or new identity was generated.

## Boundary

1450M and 1490M remain frozen. This repair does not authorize resuming them. Any successor must use a fresh identity, record stdout/stderr, verify 15 committed views, and audit identities before rechunking or training. `locked_test_used=false`.

## Process-lifetime hardening (2026-08-03)

P3o-M exposed a second, distinct operational issue after the disk-space repair: a long-lived MATLAB process that sequentially generated several power views could terminate during a later HDF5 transaction even when C: and D: had ample free capacity. The earlier views remained atomically committed; the interrupted view remained an exact-name `.partial.mat` and was never treated as data.

## Regression verification (2026-08-03)

- MATLAB R2024b parsed the repaired wrapper successfully (`P3OK_PARSE_CHECK_PASS`).
- A no-write invocation of `generate_p3o_low_power_attribution_data` completed all five regimes and three power manifests with the explicit D: output root (`P3OK_ACCESS_PREFLIGHT_PASS`).
- The one-view P3o-Q wrapper also completed its no-write manifest and output write-probe (`P3OQ_ACCESS_PREFLIGHT_PASS`). Its return value is a scalar struct, rather than a cell; this was verified without treating the return-type difference as an I/O error.
- `D:\\CJ0724_generated_data` contained no `*.partial.mat` after the checks. At verification time its free capacity was 14.65 GiB, so a full 15-view run through the legacy wrapper is deliberately rejected by its 15 GiB guard before HDF5 output starts. Single-view wrappers retain their separate 5 GiB guard.

The durable execution rule is now one `regime × power` train transaction per MATLAB process, with the existing exact-name `resume` policy. A failed view is resumed only after the process exits; its own `.partial.mat` is cleaned by the source generator before regeneration, while previously committed views are enhanced-read verified and reused. P3o-M completed all 30 fresh train/development views under this rule with zero residual partial files. This changes process lifecycle only, not seeds, waveforms, parameter ranges, sample counts, or identity contracts.
