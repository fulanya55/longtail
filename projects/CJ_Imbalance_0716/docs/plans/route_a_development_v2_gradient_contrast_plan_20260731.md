# Route A paired-gradient contrast extrapolation preregistration

## Motivation

The paired shared-representation experiment produced positive M5-minus-C6 target
weak-recall deltas in all five seeds, positive macro-AUPR and both SawFM-context
subgroups, but missed the frozen effect-size gate.  This follow-up tests whether
the reproducible paired update direction contains a stronger conflict-removal
solution beyond the endpoint reached by ordinary M5 optimization.

## Candidate

For each seed, let `theta_C6` and `theta_M5` be the already frozen 12-epoch paired
checkpoints.  Define:

`theta(alpha) = theta_C6 + alpha * (theta_M5 - theta_C6)`.

Only floating-point tensors are interpolated; integer buffers use the M5 value.
The fixed candidate grid is `[1.0, 1.5, 2.0, 3.0]`.  Alpha is selected on the
existing validation role by common-five macro-AUPR, tie-broken by mean
MFT/VFT/ISRJ weak-positive-versus-negative AUROC.  Alpha=1 is exactly M5, so the
selection cannot be forced to extrapolate.

The selected checkpoint is calibrated on calibration, thresholded on
threshold-selection, and evaluated once on early-stop.  The consumed historical
locked test and sealed test-v2 remain unopened.

## Gate

The selected contrast candidate is compared with the paired C6 endpoint using
the unchanged eight H5/shared-gradient promotion checks: macro-AUPR >= -0.005,
macro-F1 >= +0.010, target weak recall >= +0.030, each target AUPR >= -0.010,
target FPR <= 0.20, positive weak-recall delta in at least 4/5 seeds,
SawFM-present macro-F1 > 0, and SawFM-absent macro-F1 >= -0.005.

Passing is development-v2 validation of the conflict-removal innovation and
only authorizes preregistration of a one-time independent test-v2 confirmation.
