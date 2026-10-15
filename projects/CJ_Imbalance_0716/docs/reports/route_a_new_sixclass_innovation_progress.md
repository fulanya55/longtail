# Route-A 新六类创新实验进度（截至 2026-09-04）

## 数据与冻结边界

当前分支为 `codex/route-a-ama-sixclass-baseline-20260826`。目标六类为 FMB、AMA、SawFM、SinFM、MFT、ISRJ；SN、VFT、RangeFakeTarget 永不投影为负样本。主骨干固定为 F2_DFMS_Full/dfms_iq，输入 `[B,2,3000]`。E5 及其历史结果永久保留；本阶段所有实验 `locked_test_used=false`，尚未获得 locked-test 读取授权。

## 阶段状态

| 阶段 | 状态 | 当前结论 |
|---|---|---|
| P0 C1 纯六类基线 | 已验证 | 30 dB 主验证/开发性能接近饱和，但跨功率 10–20/45–50 dB 组合率明显下降。 |
| P1a 特征机制 | 已完成开发筛查 | anti-alias 有小幅优势，尚无稳定晋级证据。 |
| P1b I/Q 分组与容量匹配 | 已完成开发筛查 | 分组卷积成本低，但组合指标明显弱于 F2。 |
| P2a 功率均衡采样 | 失败 | 全功率混合破坏 30 dB 主边界，组合率大幅下降。 |
| P2b 邻近功率 curriculum | 失败 | 低功率有所改善，但高功率和组合稳定性不足。 |
| P2c 可见性辅助 | 部分支持 | 可见性 BCE 显著下降，组合率小幅上升，尚未证明完整功率鲁棒性。 |
| P3 双轴长尾损失 | 不晋级 | class-balanced/dual-axis 加权未稳定改善组合率，ISRJ FPR 上升。 |
| P3c 同身份跨功率一致性 | 五种子部分支持 | 验证组合率均值提升 0.234 个百分点，但 Wilcoxon p=0.072；35–50 dB 有改善，10–20 dB 下降，不满足全功率晋级。 |
| P3d 分层功率一致性门控 | 3-seed 部分支持 | high-only 改善 35–50 dB、low-only 改善 10–25 dB，但各自牺牲另一侧，证明存在功率层梯度冲突。 |
| P3e 双侧低权重一致性 | 部分支持后停止调参 | λ=0.05 缓解高功率退化，但 10–20 dB 仍无稳定改善；不再在同一开发集扫描权重。 |
| P4 匹配困难负样本 | stopped | 小型 preflight 有改善，但严格 20 组 pair group 留出、5 seeds 下组合率仅 +0.10 个百分点且 CI 跨 0，Macro-F1 下降；HN 未通过泛化门控。 |
| P5–P9 | 未执行 | P4 严格组合组留出未通过，当前不构造专家；locked test 仍禁止读取。 |

## 当前最有价值的证据

P3 证明单纯增强尾标签/弱分量梯度不能解决困难负样本的可分性；P4 初筛表明“同功率、同复杂度、不同组合”的有界排序约束比单纯重加权更接近根因，但当前配对仍只匹配总有效 JNR。P4b 审计进一步确认当前 MAT 不含可用的 receiver/background/target 条件字段，严格反事实证据尚未形成。

## 当前结构与训练协议

F2 为真实 I/Q 复数特征提取骨干，宽度 16，约 257,511 参数。开发筛查统一使用 AdamW（lr=1e-4、weight decay=1e-4）、batch=16、2 epochs、梯度裁剪 5.0、固定 0.5 阈值和 3 seeds（2039/2040/2041）；晋级候选必须补齐至少 5 seeds，并报告 CI、配对检验和成本审计。

## 下一步

P4 matched-HN 当前协议已停止。后续只开展新的、预先定义的非专家功率门控候选：分别对低功率（10–25 dB）和高功率（35–50 dB）使用独立有界正则，并以 5 seed、逐类 recall/FPR、组合转移矩阵和成本门控。若不能同时改善两端，则保留失败边界，不进入 P5。身份审计结果见 `outputs/route_a_p4_identity_audit_20260828/identity_metadata_audit.json`。
 
## P3f addendum (2026-09-05)

PCGrad was screened as a new non-expert power-consistency candidate on the fixed F2 backbone (3 seeds, 2 epochs, paired 10--50 dB development views). Low/high consistency gradients conflicted in approximately 35--41% of mini-batches. PCGrad improved 35/40 dB combination accuracy and 45/50 dB AUPR, but reduced 10/20/25 dB AUPR and 25 dB combination accuracy; the 30 dB paired validation gain was negligible (exact-combination +0.016 pp, Wilcoxon p=0.32). It is therefore **partial support only, not promoted, and tuning is stopped**. PCGrad cost was about 1.82x BCE wall-clock with identical parameters. Evidence: `docs/reports/route_a_p3f_pcgrad_20260905.md`; locked test remains unread and P5 experts remain unconstructed.
 
## P3g addendum (2026-09-05)

The low-power visibility auxiliary candidate was screened on the fixed F2 backbone with three seeds. It raised low-power AUPR in some bands but destabilized the 30 dB classification anchor: mean exact-combination accuracy fell by 5.60 percentage points, with ISRJ/MFT false-positive rates reaching 0.1301/0.0309 and 0.1111/0.0689 in seeds 2040/2041. P3g is therefore **failed and stopped**. The evidence indicates unrestricted shared-backbone visibility gradients are not equivalent to class-discriminative gradients. P4 and P5 remain stopped; locked test remains unread. Full report: docs/reports/route_a_p3g_low_visibility_aux_20260905.md.
 
## P3h addendum (2026-09-05)

Anchor-projected low-power visibility supervision was screened after P3g. The visibility gradient opposed the 30 dB classification gradient in 74--96% of mini-batches, but projection did not prevent shared-representation drift: mean 30 dB exact-combination accuracy fell by 5.84 percentage points, with ISRJ/MFT FPR up to 0.1421/0.0778. P3h is **failed and stopped**. Direct and projected shared-backbone visibility objectives are now both negative evidence; no weight sweep or expert construction is authorized. Full report: docs/reports/route_a_p3h_anchor_projected_visibility_20260905.md.
 
## Frozen F2 feature audit addendum (2026-09-05)

The corrected audit fully loaded the C1 checkpoint, including the classifier head. Within-power four-fold linear probes show low exact-combination separability at 10--20 dB (0.285--0.633), but high separability at 25--40 dB (0.930--1.000) and still useful information at 45--50 dB. The fixed 30 dB head transfers poorly at 25--50 dB despite separable frozen features, while 10--20 dB has a genuine observability limit. The first audit output was invalid because it skipped classifier weights and is retained with an INVALID_RUN notice; only the v2 output is evidence. This closes the shared visibility-auxiliary line and redirects the next experiment toward a physically justified low-power front-end plus separately preregistered cross-power calibration, without experts or locked-test access.
 
## Frozen F2 scalar-calibration addendum (2026-09-05)

The full-checkpoint scalar-logit calibration audit separates the current failure: 30 dB-fitted calibration does not improve cross-power combination accuracy, while within-power diagnostic calibration raises 20/25/35/40/45 dB but cannot recover 10/15 dB and is unstable at 50 dB. Calibration can therefore explain mid/high-power score mismatch but not the low-power observability limit. No calibration module is promoted; P3g/P3h visibility lines, P4/P5, and locked-test access remain stopped.
 
## P3i addendum (2026-09-05)

The normalized complex phase-increment input screen failed the fixed 30 dB anchor: mean exact-combination accuracy decreased by 18.95 percentage points and 10--25 dB AUPR also declined. Some 40--50 dB metrics improved, but the representation discards amplitude/Cartesian evidence and is noise-sensitive at low SNR. P3i is stopped; raw normalized I/Q remains the only accepted F2 input. Report: docs/reports/route_a_p3i_phase_increment_20260905.md.
 
## P3j addendum (2026-09-05)

The raw-I/Q plus bounded phase residual screen preserved the 30 dB anchor but the residual scale stayed below 3e-4 in all seeds, so the effective residual contribution was below 3e-5. Low-power combination accuracy did not improve; the tiny apparent anchor gain was one decision in one seed. P3j is therefore a null-increment failure, not a promoted innovation. Raw normalized I/Q remains the accepted F2 input and no further bound/initialization tuning is authorized on this split.
 
## P3k addendum (2026-09-05)

The pooled 10--50 dB scalar-logit calibrator improved 25/40/45 dB combination decisions but reduced the fixed 30 dB validation combination accuracy from 1.0000 to 0.9736 and did not improve 10--20 dB. It is a calibration-only partial diagnostic, not a promoted candidate; affine tuning is stopped. The evidence separates low-power information loss from mid/high-power score-geometry mismatch.
 
## P3l addendum (2026-09-05)

The fixed-margin (2.0) anchor-safe pooled calibration gate activated on 0% of all development samples. It preserved the 30 dB anchor only by becoming exactly the raw F2 model, with no low-power or mid-power gain. P3l is a null-effect diagnostic and margin tuning is stopped; P3k/P3l do not justify a promoted calibration module.
 
## Evidence-audit addendum (2026-09-05)

The current machine-readable ledger is outputs/route_a_innovation_failure_ledger_20260905.json and the consolidated audit is docs/reports/route_a_innovation_evidence_audit_20260905.md. No candidate has passed all joint gates. The evidence now separates a genuine 10--15 dB observability limit from a 20--50 dB cross-power score-geometry problem. No shared visibility, phase replacement/residual, or pooled calibration mechanism is promoted; P4/P5 remain stopped and locked-test access remains sealed.
 
## P1a five-seed confirmation addendum (2026-09-06)

The anti-aliased F2 variant completed the planned five-seed confirmation. It improves 30 dB exact-combination accuracy by 0.586 percentage points on average and gives the clearest 35--50 dB robustness gains, with unchanged parameter count. However, paired Wilcoxon p=0.068 for exact combination and a small 20 dB AUPR decrease prevent a strict confirmatory/final claim. Anti-alias is retained as the strongest development candidate, not frozen; P4/P5 and locked-test access remain stopped.
 
## A-final-prelock addendum (2026-09-07)

Anti-aliased downsampling completed a second independent five-seed screen. Across two splits, the mean 30 dB exact-combination gain is approximately +0.684 percentage points with unchanged parameter count; the single-split Wilcoxon values remain borderline (0.0625--0.0679). It is frozen only as A-final-prelock with explicit limitations, source checkpoint hashes, and no locked-test access. The prelock manifest is frozen/route_a_anti_alias_Afinal_prelock_20260907/manifest.json; locked-test reading still requires explicit authorization.

## P1a cross-split synthesis addendum (2026-09-07)

The two five-seed development splits are now recorded as a ten-observation exploratory synthesis in `outputs/route_a_p1a_anti_alias_Afinal_prelock_20260907/combined_replication_statistics.json`. This file records the paired values, costs, power-wise means, and the post-hoc Wilcoxon calculations. It strengthens reproducibility of the prelock decision but is not a new confirmatory claim: the synthesis was made after observing both development splits, and low-power (15--20 dB) benefit remains unestablished.

## P9 pre-registration and prelock integrity addendum (2026-09-07)

The one-time locked-test protocol is preregistered at `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_preregistration.json`. The candidate and F2_full ablation are fixed, the threshold is 0.5, and the ten checkpoints use uniform logit averaging with no post-read selection. A read-only integrity audit verified all ten checkpoint paths and SHA256 values; `read_count=0`, `test_files_opened=false`, and `locked_test_used=false`. A data-role audit flags that the historical C-drive v3 test is out of scope and the E-drive power-sweep tests were already opened by C1; a fresh identity-disjoint E-drive sealed source must be bound before authorization. Explicit user authorization remains required before any eligible sealed test file is read.

## Locked-test source binding plan addendum (2026-09-08)

The source-binding plan is recorded in `docs/plans/route_a_new_sixclass_locked_test_binding_plan_20260908.md`. It requires a fresh E-drive identity-disjoint nine-power source before authorization. Existing development tests and Route-B generated tests are explicitly excluded; the current MATLAB entry point still depends on a historical C-drive generator path, so no generation or test read was attempted in this turn.

## Locked-test source generation and preseal audit addendum (2026-08-27)

A fresh E-drive test-only source was generated under `generated_data/route_a_locked_test_ama_seed4050000000` using the migrated configurable generator, seed `4050000000`, and the fixed six-class order `FMB, AMA, SawFM, SinFM, MFT, ISRJ`. It contains nine powers (10--50 dB in 5 dB steps), 63 combinations per power, 40 samples per combination, and 2,520 samples per power (22,680 total). SN, VFT, and RangeFakeTarget are absent from `known_label`; `label_selected` equals `known_label`; each selected class has 1,280 positives per power; unknown fields are zero.

The preseal metadata audit opened only HDF5 metadata/labels and did not read `iq_complex` for model evaluation. All 2,520 new `receiver_noise_seed` values are unique within the source and have zero intersection with the 18 existing nine-class train/test metadata files (3,348,000 receiver-seed values scanned). The audit receipt is `outputs/route_a_locked_test_source_generation_20260908/metadata_identity_audit.json`.

All nine MAT files and the sweep CSV were SHA256-recorded in `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_source_sealed_manifest.json`. The source is sealed and preregistered, but no model inference or locked-test metric has been run. `locked_test_used=false`; explicit user authorization is still required for the one-time read. The failed detached launch (PID 28860) is retained; the foreground generation receipt records successful completion and the metadata-only audit.

## Locked-test execution freeze addendum (2026-08-27)

The one-time runner is now frozen at `experiments/run_route_a_locked_test_once.py` (SHA256 `f3429007ec31d41ee20cd2970123144938fa9379754b21c1ef0eb1a6440f048c`). The execution contract is `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_execution_freeze.json` (SHA256 `a7530c77f0fdf063fb00b0b17ca39739d5856b9d09b4e638a2c1b89eb34a68f0`) and binds ten anti-aliased candidate checkpoints to ten paired F2_full checkpoints; all 20 checkpoint hashes match. The runner verifies the sealed-source and candidate-manifest hashes after authorization, requires CUDA, refuses a non-empty output root or repeat terminal receipt, emits raw JSON before the compact Markdown report, and forbids post-read changes. A first authorized attempt aborted before model inference because PyTorch required explicit CUDA device initialization; the abort receipt is retained, and only this operational initialization fix was made. No test metrics existed at that point; the retry uses the same frozen data/models/thresholds.

## A-final one-time locked-test result addendum (2026-08-27)

After explicit user authorization, the frozen runner completed exactly one read of the fresh E-drive identity-disjoint source. The anti-aliased candidate versus paired F2_full ten-checkpoint ensemble achieved Macro-AUPR `0.929629` vs `0.927379` (+0.225 pp), Macro-F1 `0.803479` vs `0.801324` (+0.216 pp), Hamming loss `0.166755` vs `0.168335`, and six-class combination exact accuracy `0.382099` vs `0.376455` (+0.564 pp). Paired checkpoint Wilcoxon p-values were 0.003906 (Macro-AUPR), 0.027344 (Macro-F1), and 0.027344 (combination exact accuracy). Improvements concentrate at 25/35/40 dB; 10/15 dB exact-combination accuracy is unchanged at 0.221429, and MFT AUPR/recall are slightly lower. Therefore anti-aliasing has positive but modest locked-test evidence and does not solve low-power observability or all-class robustness. Full result: `docs/reports/route_a_locked_test_result_20260827.md`; raw JSON and terminal receipt are under `outputs/route_a_locked_test_once_20260827`. The locked source is now consumed and must not be used for tuning.
