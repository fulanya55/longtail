# Latest multi-label long-tail baseline reproducibility audit

## Audited reference

- Chen et al., *Towards Gradient Equalization and Feature Diversification for
  Long-Tailed Multi-Label Image Recognition*, IEEE TMM, 2025,
  DOI `10.1109/TMM.2025.3535395`.

Its abstract describes three coupled elements: group-wise balanced sampling,
dynamic gradient balancing loss, and a category-information diversity module.
This makes it a relevant modern comparator for a long-tailed multi-label
claim, but not a drop-in loss function.

## Reproducibility check on 2026-08-01

| Check | Observation | Consequence |
|---|---|---|
| Bibliographic identity | Crossref confirms the DOI, authors, journal, and 2025 publication | Reference is real and relevant |
| Method specification | Semantic Scholar record supplies only the abstract-level description | Insufficient to reproduce the coupled sampling/loss/module algorithm faithfully |
| Full text access | Semantic Scholar reports `CLOSED`, with no open PDF URL | Cannot derive unreported equations or schedules from an accessible primary source |
| Public code search | GitHub repository search for exact title returned zero repositories | No public exact-title implementation located in the reproducibility check |
| Equal-protocol runner | None available | No fair numerical comparison may be reported |

## Decision

Do **not** implement an approximation under the GEDFD name and do **not** mark
the latest-method baseline requirement complete.  An ad-hoc implementation
would confound three unstated choices—group construction, gradient update
schedule, and diversity-module architecture—and could make a favorable or
unfavorable number irreproducible.

If author code or a full algorithm description becomes available, the required
implementation protocol is: preserve six-class manifests and receiver-seed
roles; match the Focal-36 total epoch budget; perform calibration and threshold
selection only on their assigned roles; run all five seeds; and use a fresh
development-v13 confirmation after selecting a genuinely changed candidate.

This audit itself did not access any evaluation data and did not use test-v3.

## Official API refresh

After the original audit, a second read-only refresh against Crossref and the
GitHub repository-search API confirmed the same title, DOI, and six author
family names (`Chen, Cui, Zhang, Deng, Xia, Lu`), while the exact-title GitHub
repository query again returned zero results. This reinforces the evidence
boundary above; it does not turn the absence of a repository into proof that
private author code does not exist.

Receipts: `outputs/tapc/route_a_latest_mllt_reproducibility_audit/public_code_refresh_20260801.json` and the stronger four-query refresh `outputs/tapc/route_a_latest_mllt_reproducibility_audit/public_code_refresh_20260803.json`.

## 2026-08-03 public-candidate triage

A broader source-level triage found two newly public repositories that describe
recent long-tailed multi-label methods. Neither is admitted as an equal-protocol
numeric comparator: TEC-Graph has no traceable primary publication identity or
license and is built around image visual-anchor/cache paths; RuralTail-MLR/TALC
has an inspectable Logit-level loss but its repository CFF lacks a DOI or
traceable paper identity and its public protocol remains image-carrier and
frequency-group based. The full receipt and boundary are in
`docs/reports/route_a_public_recent_mllt_candidate_triage_20260803.md`.
