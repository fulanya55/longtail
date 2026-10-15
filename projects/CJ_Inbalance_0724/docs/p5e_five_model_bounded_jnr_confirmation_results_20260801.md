# P5e: five-model confirmation of the frozen bounded JNR-only attachment

## Scope and protocol

P5e is evidence completion for the already frozen P5b3/P5b4 mechanism, not a new architecture.  It adds P4c source-model seeds 3044 and 3045 (same data-role hashes and fixed profile) to the prior seeds 3041--3043, then trains their corresponding fixed 64-hidden posterior-conditioned analytic JNR heads.  The two added development heads pass every frozen P5e gate.

The five-model candidate was then frozen before generating two new train-only paired-power identity pools: `950000000` and `960000000`.  Each pool contains 3,780 identities, all 63 combinations, and 10/15/20/25/30 dB paired views.  Both independent audits verify zero overlap with root/P4c/P5b--P5d identities, equal identity and label order across powers, exact swept-JNR parameter contract, and no test generation/opening/use.

## Confirmation result

All five models pass every pre-registered confirmation gate in both new identity pools:

| target | 950M mean ± sample SD oracle NMAE reduction | 960M mean ± sample SD oracle NMAE reduction | minimum model gain across both pools |
|---|---:|---:|---:|
| MFT.JNR | 52.79% ± 1.32% | 52.90% ± 1.32% | 50.93% |
| VFT.JNR | 53.71% ± 1.81% | 53.84% ± 1.67% | 51.57% |
| ISRJ.JNR | 45.91% ± 2.53% | 46.37% ± 2.80% | 42.47% |

For MFT and ISRJ, every model on both pools has a positive oracle-gated gain, a non-worse predicted-gated NMAE, and an identity-clustered 95% bootstrap lower bound above zero.  The attachment has no classification-logit or threshold write path; non-target parameter invariance is verified on every run.  No target has a 10 or 30 dB NMAE increase above 0.03.

## Required boundary reporting

- This is a **JNR-only** improvement.  It does not demonstrate a new classification improvement, all-12-parameter improvement, composition-disjoint classification generalization, or parameter-OOD reliability beyond the separate P5c atlas.
- The worst per-combination cell is the all-three-swept-component combination (binary ID 7): MFT has a +0.03629 NMAE change for source seed 3044 on identity pool 960M.  This is not a violation of P5e's pre-registered per-power gate, but it prevents an unqualified per-combination safety claim.
- The current confirmation implementation evaluates the small head densely and records that only about 42.1% of the three target cells are eligible for attachment.  This is **activation coverage, not realized lazy-expert acceleration**.  A separate fresh-identity, output-equivalence-preserving lazy-execution benchmark is required before claiming computational savings.

## Decision

The five-model JNR-only attachment is frozen as fresh-confirmed closed-set/in-range (10--30 dB) parameter evidence.  It is eligible for a later separately pre-registered, explicitly authorized once-only new locked test.  It must not be altered, tuned, or evaluated again on the two P5e confirmation pools.

## Artifacts

- `configs/p5e_five_model_seed_bounded_jnr_preregister_20260801.json`
- `configs/p5e_five_model_seed_confirmation_freeze_20260801.json`
- `outputs/p5e_five_model_seed_bounded_jnr_20260801/head_extension/result.json`
- `outputs/p5e_five_model_seed_bounded_jnr_20260801/audit_seed950000000.json`
- `outputs/p5e_five_model_seed_bounded_jnr_20260801/audit_seed960000000.json`
- `outputs/p5e_five_model_seed_bounded_jnr_20260801/five_model_confirmation/result.json`
