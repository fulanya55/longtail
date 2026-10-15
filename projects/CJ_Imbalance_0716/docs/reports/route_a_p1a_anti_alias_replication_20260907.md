# Route-A P1a anti-alias independent-split replication (2026-09-07)

## Status

**Replicated promising development candidate; not locked and not a final claim.** This run used a new split seed (260907), five seeds, and train-only unseen-identity development views. No locked test was opened and E5 was not modified.

## Protocol

Only F2_full (dfms_iq) and F2_anti_alias (dfms_iq_anti_alias) were compared. Input, BCEWithLogits loss, AdamW (1e-4/1e-4), batch 16, two epochs, threshold 0.5, fit/validation counts, and development-power sampling were held fixed. Each variant has 257,511 parameters.

## Results

On the new split, anti-alias improves validation exact-combination accuracy by 0.781 percentage points on average (five paired seeds), Macro-F1 by 0.137 pp and Macro-AUPR by 0.0068 pp. Wilcoxon p-values are 0.0625, 0.0625 and 0.125 respectively; this is replication evidence, not strict p<0.05 confirmation.

At 10 dB the mean AUPR change is +0.11 pp but combination accuracy is -0.08 pp. At 20 dB AUPR decreases -0.12 pp while combination accuracy is essentially unchanged. The strongest robust gains are 25 dB combination (+1.25 pp), 40 dB combination (+1.80 pp) and 50 dB AUPR (+0.69 pp). ISRJ recall rises about 0.66 pp with a mean FPR decrease of 0.08 pp; MFT FPR rises only 0.06 pp.

## Decision

Together with the previous five-seed split, anti-alias is the strongest Route-A development candidate and can be placed in an A-final-prelock manifest. It still must not be described as a locked-test result or universally power-robust innovation: the 20 dB AUPR trade-off and borderline single-split significance remain explicit limitations. The candidate is frozen for reproducibility at the manifest level; no further tuning on these splits is authorized. P4/P5 and locked-test access remain stopped.

## Artifacts

* Summary: outputs/route_a_p1a_anti_alias_independent_20260907/p1a_mechanism_summary.json.
* Statistics: outputs/route_a_p1a_anti_alias_independent_20260907/p1a_anti_alias_independent_statistics.json.
* Role manifest: outputs/route_a_p1a_anti_alias_independent_20260907/role_manifest.json.
* Terminal receipt: outputs/route_a_p1a_anti_alias_independent_20260907/terminal_receipt.json.
