# Route-A innovation evidence audit (2026-09-05)

## Scope

This audit consolidates the new six-class branch: FMB, AMA, SawFM, SinFM, MFT and ISRJ. SN, VFT and RangeFakeTarget remain excluded. The fixed contract is F2_DFMS_Full/dfms_iq with true I/Q input [B,2,3000]. E5 remains untouched. Every post-C1 experiment in this ledger records locked_test_used=false; P5 experts were never constructed.

## Evidence state

The complete machine-readable ledger is outputs/route_a_innovation_failure_ledger_20260905.json. It distinguishes verified development, partial support, failed/stopped and unexecuted stages. The C1 receipt reports test_files_opened=true but locked_test_used=false; this is historical ordinary test-role evidence and is not reinterpreted as a locked-test read.

## What is currently supported

* F2 raw normalized I/Q is a strong 30 dB closed-set classifier.
* Low/high power consistency gradients conflict; PCGrad reduces conflict but trades low-power AUPR for high-power gains.
* Frozen F2 features are intrinsically weak at 10--15 dB, while 25--45 dB retains substantial separability.
* Cross-power scalar calibration can improve selected mid-power decisions in an upper-bound diagnostic, but pooled calibration harms the 30 dB anchor and a fixed safety gate becomes a no-op.

## What is not supported

No tested candidate simultaneously preserves the 30 dB anchor, improves 10--20 dB, improves high-power generalization, and passes per-class FPR/recall and cost gates. Shared visibility auxiliary (direct or anchor-projected), pure phase replacement, bounded phase residual, pooled calibration and anchor-safe calibration are not valid innovations. P4 strict group holdout did not support matched-hard-negative generalization, so P5 expert construction remains prohibited.

## Mechanistic conclusion

The evidence separates two bottlenecks. At 10--15 dB, the frozen representation itself lacks reliable class evidence; calibration or an auxiliary visibility label cannot create it. At 20--50 dB, score geometry shifts across power despite separable features, so a future calibration/alignment intervention would need an explicit anchor-preserving, power-agnostic gate. These are separate research questions and must not be folded into one shared loss.

## Required next action

Before any additional model experiment, preregister one physically justified observation/data intervention for the 10--15 dB regime or formally close Route-A innovation claims at the current evidence boundary. If a new candidate is authorized, use three seeds for screening, five seeds only after passing the anchor and per-class gates, retain all receipts/SHA256/cost audits, and keep locked test sealed until explicit authorization.
