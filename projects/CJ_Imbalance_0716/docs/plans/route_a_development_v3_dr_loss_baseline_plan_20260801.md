# ECCV 2024 DR-Loss fair baseline

Test whether the public ECCV-2024 DR-Loss improves six-label long-tail ranking
under exactly the Focal-36 development-v3 protocol. The sole changed mechanism
is the official DR logits/labels loss with `gamma1=gamma2=1`; its per-label
reduction is applied to the existing common-five plus `0.2 * SawFM` objective.

F2 initialization and trainable scope, 36-epoch/8-patience budget, AdamW,
learning rates, batch order, frozen RawSaw route, Platt calibration role, and
per-label FPR<=0.20 threshold role all match Focal-36. Five identity-isolated
seeds are `2039..2043`; development-v3 SHA must equal
`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`.
The comparison is DR minus stored Focal-36. No hyperparameter search,
candidate selection, test-v2, locked-test, or test-v3 access is permitted.

The result is a mainstream baseline only. It cannot promote VTSE-VBF; a result
in either direction only tests whether label-frequency/hard-example robustness
alone explains the observed weak-component pattern.
