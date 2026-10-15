# Route-A P1a anti-alias five-seed confirmation (2026-09-06)

## Status

**Promising partial confirmation; not frozen.** This is train-only development evidence. No locked test was opened and E5 was not modified.

## Design

The original P1a screen ranked the anti-aliased downsampling variant ahead of the fixed F2 full variant. This run completes the planned five-seed confirmation using only those two variants. All data roles, split seed, BCE loss, AdamW settings, threshold, fit/validation counts and power-development sampling are unchanged from P1a; only the anti-alias mechanism differs.

## Results

| Variant | 30 dB Macro-AUPR | 30 dB Macro-F1 | 30 dB exact combination |
|---|---:|---:|---:|
| F2 full | 0.99974509 | 0.99364989 | 0.96396484 |
| F2 anti-alias | 0.99978840 | 0.99472840 | 0.96982422 |
| Mean difference | +0.00004331 | +0.00107851 | +0.00585938 |

Paired differences are positive for all five seeds in Macro-AUPR and four of five seeds in exact-combination accuracy. The paired Wilcoxon p-values are 0.0625 (AUPR), 0.125 (Macro-F1), and 0.0679 (exact combination). The result is encouraging but does not meet a strict p<0.05 confirmatory claim.

## Power development

Anti-alias improves exact-combination accuracy at 10, 15, 20, 25, 30, 35 and 40 dB on average; the largest gains are 35 dB (+1.41 percentage points) and 40 dB (+2.50 pp). Macro-AUPR gains are stable at 40 dB (+0.58 pp), 45 dB (+0.99 pp) and 50 dB (+0.84 pp). The 20 dB Macro-AUPR decrease (-0.18 pp) and 50 dB combination change (-0.04 pp) mean that a full all-power robustness claim is not yet justified.

## Per-class and cost audit

At the 30 dB validation threshold, ISRJ recall rises by about 0.56 percentage points while its negative FPR rises by about 0.20 percentage points; FMB, SawFM, SinFM and MFT FPRs decrease slightly or remain unchanged. Both variants have exactly 257,511 parameters. Mean training time is 13.91 s (full) versus 14.31 s (anti-alias), and measured single-sample latency is noisy with no confirmed overhead.

## Decision

Retain anti-alias as the strongest development candidate, but do not call it a final innovation or use it for locked-test selection yet. The evidence supports a modest anti-aliasing benefit, particularly in 35--50 dB robustness, while statistical significance and the 20 dB trade-off remain unresolved. A later preregistered confirmation should specify the strict gate, including 10--20 dB non-degradation and ISRJ FPR, before any final freeze. P4 hard-negative matching and P5 expert construction remain stopped.

## Artifacts

* Script: experiments/run_route_a_p1_mechanism_screen.py (P1A_VARIANTS=F2_full,F2_anti_alias).
* Summary: outputs/route_a_p1a_anti_alias_confirm_20260906/p1a_mechanism_summary.json.
* Statistics: outputs/route_a_p1a_anti_alias_confirm_20260906/p1a_anti_alias_confirm_statistics.json.
* Role manifest: outputs/route_a_p1a_anti_alias_confirm_20260906/role_manifest.json.
* Terminal receipt: outputs/route_a_p1a_anti_alias_confirm_20260906/terminal_receipt.json.
