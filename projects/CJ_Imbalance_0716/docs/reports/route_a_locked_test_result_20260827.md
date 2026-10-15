# Route-A A-final locked-test result

## 1. Protocol and provenance

The one-time locked test was authorized explicitly and completed on the fresh E-drive identity-disjoint source. The source contains six classes (`FMB`, `AMA`, `SawFM`, `SinFM`, `MFT`, `ISRJ`), nine powers (10--50 dB in 5 dB steps), 2,520 samples per power, and 22,680 samples in total. The candidate is `F2_anti_alias / dfms_iq_anti_alias`; the paired ablation is `F2_full / dfms_iq`. Each side uses the same ten prelocked checkpoints (five confirmation plus five independent), uniform arithmetic mean of logits, true normalized I/Q input `[B,2,3000]`, and threshold 0.5. No threshold, checkpoint, split, preprocessing, aggregation, or metric definition was changed after the read.

The first authorized process attempt aborted before model inference because this workstation's PyTorch build rejected `reset_peak_memory_stats(torch.device('cuda:0'))`; the abort receipt is retained. The retry only initialized the selected CUDA device before the same no-argument memory-statistics call. It completed on the RTX 5080. `read_count=1`, `locked_test_used=true`.

Authoritative artifacts:

- Raw result: `outputs/route_a_locked_test_once_20260827/raw_locked_test_result.json`
- Generated report: `outputs/route_a_locked_test_once_20260827/locked_test_report.md`
- Terminal receipt: `outputs/route_a_locked_test_once_20260827/terminal_receipt.json`
- Execution freeze: `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_execution_freeze.json`

## 2. Pooled ten-checkpoint ensemble

| Metric | F2_anti_alias | F2_full | Difference |
|---|---:|---:|---:|
| Macro-AUPR | 0.929629 | 0.927379 | +0.002250 (+0.225 pp) |
| Macro-AUROC | 0.898086 | 0.894773 | +0.003313 (+0.331 pp) |
| Macro-F1 | 0.803479 | 0.801324 | +0.002155 (+0.216 pp) |
| Hamming loss | 0.166755 | 0.168335 | -0.001580 |
| Six-class combination exact accuracy | 0.382099 | 0.376455 | +0.005644 (+0.564 pp) |
| Label-cardinality error | 0.996384 | 1.005423 | -0.009039 |
| ECE-15 | 0.110094 | 0.109587 | +0.000507 |

The ten paired checkpoint values give Wilcoxon two-sided p-values of 0.003906 for Macro-AUPR, 0.027344 for Macro-F1, and 0.027344 for combination exact accuracy. These are paired-checkpoint diagnostics, not independent test-seed replication; the locked source itself is evaluated only once.

## 3. Per-power ensemble results

| Power | Candidate Macro-AUPR | Baseline Macro-AUPR | Δ AUPR | Candidate combo exact | Baseline combo exact | Δ combo |
|---:|---:|---:|---:|---:|---:|---:|
| 10 dB | 0.846303 | 0.844021 | +0.002282 | 0.221429 | 0.221429 | +0.000000 |
| 15 dB | 0.872450 | 0.871585 | +0.000865 | 0.221429 | 0.221429 | +0.000000 |
| 20 dB | 0.936068 | 0.936216 | -0.000148 | 0.222619 | 0.221825 | +0.000794 |
| 25 dB | 0.996674 | 0.996285 | +0.000390 | 0.454762 | 0.445238 | +0.009524 |
| 30 dB | 0.999907 | 0.999898 | +0.000008 | 0.986508 | 0.982540 | +0.003968 |
| 35 dB | 0.999356 | 0.998969 | +0.000387 | 0.645635 | 0.617063 | +0.028571 |
| 40 dB | 0.970896 | 0.965796 | +0.005100 | 0.243651 | 0.236508 | +0.007143 |
| 45 dB | 0.890343 | 0.885904 | +0.004439 | 0.221429 | 0.221032 | +0.000397 |
| 50 dB | 0.838180 | 0.833952 | +0.004228 | 0.221429 | 0.221032 | +0.000397 |

The main gains occur at 25, 35, and 40 dB. The 10--15 dB exact-combination floor is unchanged, so the result does not support a claim that anti-aliasing solves the low-power observability limit. At 50 dB, AUPR improves but exact-combination accuracy remains near the same floor, indicating score/ranking improvement without reliable multi-label decoding.

## 4. Per-class pooled results

| Class | Δ AUPR | Δ recall | Δ FPR | Δ F1 |
|---|---:|---:|---:|---:|
| FMB | +0.005464 | +0.009375 | -0.000090 | +0.006435 |
| AMA | +0.001462 | +0.004253 | +0.000000 | +0.002959 |
| SawFM | +0.005666 | +0.000608 | -0.000538 | +0.000693 |
| SinFM | -0.000450 | +0.001389 | -0.000090 | +0.001013 |
| MFT | -0.001671 | -0.001649 | +0.000090 | -0.001258 |
| ISRJ | +0.003030 | +0.004167 | +0.000090 | +0.003089 |

Anti-aliasing helps FMB, SawFM, and ISRJ most clearly. MFT and SinFM remain the weak/neutral classes, with MFT slightly worse in both AUPR and recall; this is an important limitation rather than evidence of uniform class improvement.

## 5. Cost and final evidence boundary

- Elapsed time: 7,113.23 s (approximately 118.6 min).
- Candidate and baseline parameters: 257,511 each.
- Reported CUDA peak allocated memory: 246,057,984 bytes.
- GPU: NVIDIA GeForce RTX 5080; CUDA was required and used.
- Terminal receipt records `read_count=1`, `locked_test_used=true`, and `post_read_changes_prohibited=true`.

Conclusion: the frozen A-final anti-aliased downsampling mechanism has positive locked-test evidence against the matched F2_full ablation for pooled ranking, macro-F1, Hamming loss, and combination exact accuracy. The effect is modest overall (+0.564 pp exact combination) and concentrated in mid/high powers; low-power observability and MFT/SinFM separability remain unresolved. This is the final locked comparison under the current preregistration. Further model or threshold tuning would require a new, separately authorized experiment and must not reuse this locked source as development data.
