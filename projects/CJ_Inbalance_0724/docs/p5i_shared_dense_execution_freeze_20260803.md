# P5i shared-dense execution freeze

P5i used fresh train-only 3950M, 3960M and 3970M paired-power pools, each audited with zero overlap against 2,422 local historical training files.  The runner passed its preflight and entered the first frozen model's reference timing loop.

At the first guard checkpoint, it rejected a transient `conda-script.py shell.powershell hook` helper.  This helper is a non-computing Conda shell process, but was absent from the original explicit allowlist.  No `result.json` was written and no complete model comparison, latency summary, confirmation, test generation, test opening or locked-test use occurred.

The whitelist is now limited to the two explicit non-computing Conda shell subcommands (`activate` and `hook`) and remains rejecting for every other external Python process.  P5i identities remain frozen and cannot be reused.
