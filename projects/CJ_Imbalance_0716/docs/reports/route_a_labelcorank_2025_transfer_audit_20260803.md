# Route A: LabelCoRank (JAIR 2025) traceability and transfer-boundary audit

Date: 2026-08-03  
Nature: read-only literature, publisher-metadata, and public-source audit. No
0716 I/Q, labels, predictions, checkpoints, locked test, test-v2, or sealed
test-v3 contents were read.

## Conclusion

[LabelCoRank](https://doi.org/10.1613/jair.1.16968) is a traceable 2025
peer-reviewed long-tailed multi-label method and its article explicitly links
the public implementation at [821code/LabelCoRank](https://github.com/821code/LabelCoRank).
It closes the earlier **source-location** gap for one current external
mechanism. It does **not** close the equal-protocol reproduction gap and does
not authorize a new training run or test-v3 access.

The source is a BERT multi-label **text** classifier. Its reranking consumes a
dataset-specific, precomputed label co-occurrence matrix; the repository does
not release the construction of that matrix. Consequently, direct numerical
comparison in the present complex-I/Q F2 protocol would replace both the
backbone and the input interface and would not be an exact reproduction.

## What was verified from the publisher and source

| Item | Auditable observation | Interpretation |
| --- | --- | --- |
| Paper identity | DOI `10.1613/jair.1.16968`; JAIR 83; Yan Yan, Junyuan Liu, Bo-Wen Zhang | Primary publication identity is traceable. |
| Author-linked code | The paper abstract points to `https://github.com/821code/LabelCoRank`; the repository has `train/model6.py`, `train/model_train.py`, and `train/load_data.py` | The released code can be inspected as an implementation witness. |
| Core mechanism | `cls_pre` produces initial logits; selected high-ranked labels are expanded through `matrix_labels_...`; label queries attend to the input sequence; `final_output` is trained jointly with `cls_pre` | The relevant external idea is a two-stage, training-derived label-relation reranking mechanism. |
| Source interface | `AutoModel.from_pretrained`, `tokenizer.encode_plus`, and a precomputed `matrix_labels_...npy` are explicit in the code | It is not an F2 complex-I/Q interface. |
| License | GitHub metadata reports no SPDX license | Do not copy source code into this repository. |

## Admission decision

`LabelCoRank` may be cited as a recent external **methodological comparator**.
It may not be called, trained, or numerically reported as an “exact
LabelCoRank reproduction” in this project.

If the broader Route-A study reaches a newly admissible protocol-level
mechanism, the only defensible comparison is a separately named
`LabelCoRank-adapted` comparator with all of the following frozen in advance:

1. Build the 6x6 co-occurrence matrix from the training role only; no
   development-calibration, threshold, OOD, locked-test, test-v2, test-v3, or
   visibility truth may enter it.
2. Specify the F2-to-sequence mapping, label-query head, number of relation
   candidates, parameter budget, optimizer budget, and five seeds before any
   training.
3. Generate a new receiver/noise-identity-isolated development source after
   that pre-registration. It must not reuse a previously consumed development
   source to tune the adaptation.
4. Compare against an equally budgeted no-relation head and a row-permuted
   co-occurrence control. A positive result then supports the external
   *label-relation mechanism*, not the title “LabelCoRank” as an I/Q method.
5. Preserve the existing prohibition on test-v3 until all candidate,
   causality, OOD, statistics, and final-bundle gates are complete.

## Relation to the current stop rule

The current Route-A ledger has already stopped repeated MFT/VFT capacity,
ROI, threshold, and multi-PRI searches because the original physical
observability gates did not pass. LabelCoRank supplies no independent physical
observable; it is a generic label-relation decision mechanism. Therefore it
cannot reopen those stopped branches or justify a fresh model by itself. Its
value at the current stage is to make the external-baseline evidence boundary
more rigorous while the required non-overlapping physical observability gate
remains unsatisfied.

## Reproducibility receipt

- Audit script: `experiments/audit_route_a_labelcorank_transfer_audit.py`
- Unit test: `tests/test_route_a_labelcorank_transfer_audit.py`
- Machine receipt: `outputs/tapc/route_a_labelcorank_transfer_audit/receipt.json`
