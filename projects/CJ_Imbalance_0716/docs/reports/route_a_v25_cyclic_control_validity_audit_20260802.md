# Route A v25 cyclic-source control validity audit

## Corrective finding

The v25 cyclic-source comparison is **not a valid source-controlled test of
target-specific local evidence**. It must be retained as an exploratory
architecture screen, but its MFT, VFT and ISRJ candidate-minus-control deltas
must not be used as causal evidence or as grounds for a confirmation run.

## Why the comparison is permutation-equivalent

The three local branches in `TargetSpecificComplexEvidenceClassifier` have the
same filter-bank, mixer, evidence-head and local-classifier architecture. In
the candidate, target positions MFT/VFT/ISRJ consume branches 0/1/2. In the
cyclic control they consume branches 1/2/0, but each target's BCE loss is still
back-propagated through the branch that generates that target's output.

Consequently, if the control's branch 1/2/0 parameters are initialized as the
candidate's branch 0/1/2 parameters, respectively, both models have exactly
the same logits. A deterministic local-only BCE update preserves this
permutation relation. This executable assertion is now covered by
`tests/test_target_specific_complex_evidence_control_equivalence.py`.

The v25 runner instead copied a single ordinary random state into both models
without applying the required branch permutation. Thus its MFT candidate used
one random branch while its control used another. The observed differences may
therefore contain random branch-initialization assignment effects, even though
the parameter count, batch order and labels were otherwise matched.

## Disposition

- The historical v25 receipt remains unchanged and correctly records that its
  joint gate failed.
- Its reported MFT `+0.01877` weak-AUROC direction is no longer admissible as
  source-controlled target-specific evidence.
- No MFT-only confirmation will be generated from this result.
- A future raw-local architecture must use a control that changes information
  content or a genuinely target-distinct inductive bias, while keeping the
  initial function, capacity, optimizer and supervision contracts matched.
- The VFT/ISRJ window-state screen remains a separate physical-identifiability
  result: it failed all three gates, and it independently rules out the tested
  fixed phase-reset statistic.

No historical locked test, test-v2 or sealed test-v3 data were opened by this
audit.
