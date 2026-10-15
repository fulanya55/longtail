"""Generate the preregistered 17-section v12 FC-PTV decision report and manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/tapc/route_a_development_v12_fc_ptv"
SUMMARY_PATH = OUTPUT_ROOT / "route_a_development_v12_fc_ptv_summary.json"
REPORT_PATH = PROJECT_ROOT / "docs/results/route_a_development_v12_fc_ptv_results_20260801.md"
MANIFEST_PATH = OUTPUT_ROOT / "artifact_manifest.json"
DECISION_RECEIPT_PATH = OUTPUT_ROOT / "final_decision_receipt.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(summary, variant, metric):
    return float(summary["aggregate"][variant][metric]["mean"])


def _delta(summary, variant, metric):
    return float(summary["comparisons_vs_c6"][variant]["aggregate"][metric]["mean"])


def _ci(summary, variant, metric):
    return summary["comparisons_vs_c6"][variant]["aggregate"][metric]["bootstrap_95_ci"]


def _f(value, digits=5):
    return f"{float(value):.{digits}f}"


def _check_rows(checks):
    return "\n".join(f"| `{name}` | {'通过' if passed else '失败'} |" for name, passed in checks.items())


def build_report(summary, cost, identity, freeze):
    variants = list(summary["protocol"]["variants"])
    absolute_rows = []
    for name in variants:
        absolute_rows.append(
            f"| {name} | {_f(_mean(summary, name, 'macro_aupr'))} | "
            f"{_f(_mean(summary, name, 'macro_f1'))} | "
            f"{_f(_mean(summary, name, 'target_mean_weak_recall'))} | "
            f"{_f(_mean(summary, name, 'target_mean_negative_fpr'))} |"
        )
    fc = summary["comparisons_vs_c6"]["FC_PTV"]["aggregate"]
    full_comparison = summary["improvement_vs_fullalpha_gate"]["comparison"]["aggregate"]
    label_rows = []
    for name in ("MFT", "VFT", "ISRJ"):
        metric = fc["per_label_weak_recall"][name]
        index = ("MFT", "VFT", "ISRJ").index(name)
        lower = summary["simultaneous_target_gate"]["bounds"]["familywise_95_lower"][index]
        label_rows.append(
            f"| {name} | {_f(metric['mean'])} | [{_f(metric['bootstrap_95_ci'][0])}, {_f(metric['bootstrap_95_ci'][1])}] | {_f(lower)} |"
        )
    q_rows = []
    for seed_row in summary["per_seed"]:
        diagnostic = seed_row["fc_ptv"]
        q_rows.append(
            f"| {seed_row['seed']} | {_f(diagnostic['q_star'][0], 3)} | {_f(diagnostic['q_star'][1], 3)} | "
            f"{_f(max(max(row) for row in diagnostic['solver']['linear_distances']))} | {_f(diagnostic['exact_dmax'])} | "
            f"{_f(seed_row['orthogonal_control']['relative_match_error'])} |"
        )
    total_train_seconds = sum(
        float(training["train_seconds"])
        for seed_row in summary["per_seed"]
        for training in seed_row["training"].values()
    )
    total_checkpoints = len(list(OUTPUT_ROOT.rglob("best_model.pth")))
    total_checkpoint_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("best_model.pth"))
    total_epochs = sum(
        int(training["epochs_ran"])
        for seed_row in summary["per_seed"]
        for training in seed_row["training"].values()
    )
    optimizer_steps = total_epochs * math.ceil(1920 / 64)
    report = f"""# Route A development-v12 FC-PTV 正式结果与停止决策

## 1. 本轮最终结论

预注册结论为 **`stop_fc_ptv`**。FC-PTV 在 development-v12 上不是一个得到验证的稳定化创新点：5/5 种子的约束均不活跃，`q*=({1.5},{1.5})`，所以 FC-PTV、FullAlpha1p5 与 ParameterNormOnly 在本轮形成数值等价的参数点和同一组评价结果。候选自身门槛、相对 Full1.5 门槛、三标签同时改善门槛和机制特异性门槛均未全部通过。按停止规则，不进入 development-v13，不做鲁棒性扫描，不执行任何外部或测试集评估。

## 2. 研究问题与可证伪假设

研究问题是：能否仅依靠 C6 fit-only 功能半径，把 SawFM 标签删除产生的配对任务向量投影到一个共享层/分类器联合信赖域，在保留 Full1.5 收益的同时降低最差标签风险和跨种子波动。可证伪点包括：约束必须实际改变系数、FC 必须超过 Full1.5、三目标标签必须获得联合统计支持，并且必须超过参数范数、classifier-only 与正交功能匹配控制。

## 3. 文献差距与创新边界

预注册文献审计确认，task arithmetic、TIES/AdaMerging、Fisher/RegMean、TATR、Fisher-Rao/KL 几何和多任务梯度方法已覆盖参数合并、敏感度信赖域、功能距离或梯度冲突。因而本候选最多只能主张一个很窄的组合：**单标签 loss-deletion 配对向量 + C6 fit-only common-five 稳健 logit 半径 + shared/classifier 两块投影 + 禁止 test adaptation**。本轮约束没有激活，故这个窄创新也未形成实证支撑。

## 4. 证据边界与预注册状态

- 预注册先于 v12 正式模型评估，文档 SHA256：`65ac809176df91a1a2bc344c92b88ccada9e53af144cc9696e481732c128791c`。
- 实现冻结 SHA256：`{_sha256(PROJECT_ROOT / 'outputs/tapc/route_a_fc_ptv_implementation_freeze/implementation_freeze.json')}`。
- development-v12 只承担开发证据；历史 locked test、test-v2、sealed test-v3 和预留 test-v12 均未使用。
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`、`test_v12_used=false`。

## 5. fresh development-v12 数据身份

- train-only 样本数：{identity['sample_count']}；文件大小：{identity['size_bytes']} bytes。
- 数据 SHA256：`{identity['sha256']}`。
- sample seed 契约：`{identity['expected_sample_seed_range']}`。
- receiver_noise_seed：`{identity['observed_receiver_noise_seed_range']}`，唯一数 {identity['receiver_noise_seed_unique']}。
- 与 development-v2--v11、test-v2/v3 契约及其他保留角色均不重叠；`test_files_opened=false`。

## 6. 网络结构与可训练作用域

底座为 `F2_DFMS_Full` 六类复数 I/Q 网络：双复数多尺度编码分支，随后进行 late fusion；并行 raw temporal 多尺度分支采用卷积核 3/7/15，融合到 128 维特征，最后由 128→6 分类器输出六类 logits。完整模型含 {cost['floating_parameter_count']:,} 个浮点参数。任务向量只允许落在最后共享 `temporal_backbone.0.*` 和 `classifier.*`；其余复数编码器、融合与投影层保持 C6 锚点不变。

## 7. FC-PTV 冻结机制

每个种子从 `d=M5-C6` 构造 shared 与 classifier 两个块。C6 fit-only logits 按 common-five 标签及 SawFM-present/absent 分层，用 `1.4826×MAD` 给出半径；对每个标签/分层构造 2×2 功能 Gram 矩阵，并求解距离 `(1.5,1.5)` 最近的可行系数，约束为 `0<=q_shared<=q_classifier<=1.5` 且每个功能二次型不超过 1。精确前向 `Dmax<=1.05` 才有效。

## 8. 正式变体与公平性

正式比较包括 C6、M5、Full1.5、FC-PTV、参数范数匹配、classifier-only 2.0、模型中点、CAGrad0.4 和正交任务子空间功能匹配控制。五个其他标签删除端点只用于构造正交控制。所有方法共享初始化、split、long-tail manifest、训练角色、早停、校准、阈值、种子和可训练范围。

## 9. 正式运行完整性

正式种子为 `[2039,2040,2041,2042,2043]`，每个种子的 9 个评估变体均为 valid；数据、split、manifest、checkpoint、功能诊断和逐种子指标均已保存。正式总汇总状态为 `{summary['status']}`，证据级别为 `{summary['evidence_level']}`，没有使用 `--resume` 选择结果，也没有结果后重算系数。

## 10. 各变体绝对性能

| 变体 | macro-AUPR | macro-F1 | target weak recall | target FPR |
|---|---:|---:|---:|---:|
{chr(10).join(absolute_rows)}

## 11. FC-PTV 相对 C6 的候选效应

- macro-AUPR：`{_f(fc['macro_aupr']['mean'])}`，95% CI `[{_f(fc['macro_aupr']['bootstrap_95_ci'][0])}, {_f(fc['macro_aupr']['bootstrap_95_ci'][1])}]`。
- macro-F1：`{_f(fc['macro_f1']['mean'])}`，95% CI `[{_f(fc['macro_f1']['bootstrap_95_ci'][0])}, {_f(fc['macro_f1']['bootstrap_95_ci'][1])}]`。
- target mean weak recall：`{_f(fc['target_mean_weak_recall']['mean'])}`，95% CI `[{_f(fc['target_mean_weak_recall']['bootstrap_95_ci'][0])}, {_f(fc['target_mean_weak_recall']['bootstrap_95_ci'][1])}]`；低于预注册 `+0.025`，因此候选门槛失败。
- target FPR delta：`{_f(fc['target_mean_negative_fpr']['mean'])}`；SawFM-present macro-F1 delta：`{_f(fc['sawfm_present_macro_f1']['mean'])}`；SawFM-absent delta：`{_f(fc['sawfm_absent_macro_f1']['mean'])}`。

| 候选门槛检查 | 结果 |
|---|---|
{_check_rows(summary['candidate_effect_gate']['checks'])}

## 12. FC-PTV 相对 FullAlpha1p5

所有逐种子和逐指标差值均为 0。最差标签改善为 `{_f(summary['improvement_vs_fullalpha_gate']['worst_target_margin'])}`，target weak、macro-F1、macro-AUPR 和 FPR 的 FC-Full 均为 `{_f(full_comparison['target_mean_weak_recall']['mean'])}` 或数值等价的 0；没有正 CI 优势，跨种子标准差也未降低。

| 相对 Full1.5 检查 | 结果 |
|---|---|
{_check_rows(summary['improvement_vs_fullalpha_gate']['checks'])}

## 13. 三目标标签同时改善检验

| 标签 | FC-C6 weak recall mean | 普通 paired 95% CI | family-wise 95% lower |
|---|---:|---:|---:|
{chr(10).join(label_rows)}

三标签均值为正，且每个标签至少 4/5 种子非负；但 VFT 与 ISRJ 的 family-wise 下界分别为 `{_f(summary['simultaneous_target_gate']['bounds']['familywise_95_lower'][1])}` 和 `{_f(summary['simultaneous_target_gate']['bounds']['familywise_95_lower'][2])}`，未超过 0。因此只能说本轮均值方向非负，不能声称三个标签均获得统计可靠改善。

## 14. 机制特异性控制

- FC 与 ParameterNormOnly 完全相同，无法证明功能约束优于普通全方向缩放。
- FC 相对 ClassifierOnly2.0：target weak mean `+{_f(summary['mechanism_specificity_gate']['classifier_only_comparison']['aggregate']['target_mean_weak_recall']['mean'])}`，但 macro-F1 为 `{_f(summary['mechanism_specificity_gate']['classifier_only_comparison']['aggregate']['macro_f1']['mean'])}`，且主要 weak 指标无正 CI 优势，门槛失败。
- FC 相对正交功能匹配控制：target weak `+{_f(summary['mechanism_specificity_gate']['orthogonal_comparison']['aggregate']['target_mean_weak_recall']['mean'])}`，95% CI `[{_f(summary['mechanism_specificity_gate']['orthogonal_comparison']['aggregate']['target_mean_weak_recall']['bootstrap_95_ci'][0])}, {_f(summary['mechanism_specificity_gate']['orthogonal_comparison']['aggregate']['target_mean_weak_recall']['bootstrap_95_ci'][1])}]`；这一项通过，但不足以建立整体机制特异性。

| 机制检查 | 结果 |
|---|---|
{_check_rows(summary['mechanism_specificity_gate']['checks'])}

## 15. 功能距离、系数与失败根因

| seed | q_shared | q_classifier | 线性 Dmax | 精确 Dmax | 正交匹配相对误差 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(q_rows)}

所有线性 Dmax 仅为 0.222--0.368，精确 Dmax 仅为 0.169--0.264，远小于半径 1。因此 `(1.5,1.5)` 本身已可行，投影没有发生。根因不是求解失败，而是 **C6 MAD 半径相对于该任务向量过宽，预注册约束在当前尺度下没有辨别力**。由于不允许结果后收紧半径或重新搜索，FC-PTV 必须停止，而不是改半径再试。

## 16. 统计、计算成本与证据缺口

- 五种子 8 个训练端点/基线，共 {total_epochs} 个训练 epoch、约 {optimizer_steps} 个 optimizer step；训练函数记录的累计时间约 {total_train_seconds:.2f} s。
- 每种子额外 6 次完整 fit-anchor 模型遍历用于 base/shared/classifier/FC/orth-unit/orth-matched 功能诊断，共 30 次数据遍历。
- 保存 {total_checkpoints} 个 checkpoint，总大小 {total_checkpoint_bytes / 1024 / 1024:.2f} MiB；代表性 FC checkpoint 为 {cost['checkpoint_size_bytes'] / 1024 / 1024:.2f} MiB。
- RTX/CUDA 推理基准：batch=128，均值 {cost['latency_ms_per_batch_mean']:.3f} ms/batch，即 {cost['latency_ms_per_sample_mean']:.4f} ms/sample，吞吐 {cost['throughput_samples_per_second']:.1f} sample/s；推理峰值 allocated memory {cost['inference_peak_allocated_bytes'] / 1024 / 1024:.1f} MiB。
- 正式训练未预先埋点 peak GPU memory，因此该项标记为缺失，不能用推理峰值替代训练峰值，也不为补指标而重跑正式实验。
- 五种子 bootstrap 的样本量很小，CI 只支持 development 级证据；且 v12 与历史数据来自同一生成体系，不是外部域证据。

## 17. 决策、停止规则与下一创新方向

正式决策：**停止 FC-PTV 方向**。不生成 development-v13，不进入全 canonical baseline 阶段，不进行鲁棒性扫描，也不访问任何 test。该失败消耗一个机制周期；它证明“固定 MAD 半径的两块投影”没有实际施加约束。

若未来另立新 Goal，必须是新的可证伪机制而非继续调当前半径。较合理的问题是先验证“功能半径能否从可接受功能漂移/风险边界中获得尺度可识别性”，例如在完全新的开发身份上预注册无监督尺度校准或局部曲率归一化，并把“约束激活率”设为先验机制门槛；在机制门槛未通过前，不再做性能性实验。历史 FullAlpha1p5 仍只能作为当前最强 task-vector 结果，不能包装成 FC-PTV 创新。
"""
    return report


def build_manifest(report_sha, decision):
    files = {}
    roots = [
        PROJECT_ROOT / "outputs/tapc/route_a_fc_ptv_preregistration",
        PROJECT_ROOT / "outputs/tapc/route_a_fc_ptv_implementation_freeze",
        PROJECT_ROOT / "outputs/tapc/route_a_fc_ptv_core_smoke",
        PROJECT_ROOT / "outputs/tapc/route_a_development_v12_generation",
        PROJECT_ROOT / "outputs/tapc/route_a_development_v12_fc_ptv_smoke",
        OUTPUT_ROOT,
    ]
    for root in roots:
        for path in root.rglob("*") if root.is_dir() else []:
            if not path.is_file() or path in {MANIFEST_PATH, DECISION_RECEIPT_PATH}:
                continue
            relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            files[relative] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    selected_sources = [
        PROJECT_ROOT / "docs/plans/route_a_fc_ptv_literature_gap_and_fresh_development_preregistration_20260801.md",
        PROJECT_ROOT / "experiments/benchmark_route_a_fc_ptv_v12_cost.py",
        PROJECT_ROOT / "experiments/report_route_a_development_v12_fc_ptv.py",
    ]
    for source in selected_sources:
        files[str(source.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    files[str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {"size_bytes": REPORT_PATH.stat().st_size, "sha256": report_sha}
    return {
        "status": "completed",
        "experiment": "route_a_development_v12_fc_ptv",
        "decision": decision,
        "artifact_count": len(files),
        "files": files,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v12_used": False,
        "test_access_allowed": False,
    }


def main():
    summary = _read(SUMMARY_PATH)
    cost = _read(OUTPUT_ROOT / "cost_benchmark.json")
    identity = _read(PROJECT_ROOT / "outputs/tapc/route_a_development_v12_generation/identity_audit.json")
    freeze = _read(PROJECT_ROOT / "outputs/tapc/route_a_fc_ptv_implementation_freeze/implementation_freeze.json")
    if summary["next_decision"] != "stop_fc_ptv":
        raise ValueError("this report generator is frozen for the v12 stop decision")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(summary, cost, identity, freeze), encoding="utf-8")
    report_sha = _sha256(REPORT_PATH)
    manifest = build_manifest(report_sha, summary["next_decision"])
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    receipt = {
        "status": "completed",
        "decision": summary["next_decision"],
        "report": str(REPORT_PATH),
        "report_sha256": report_sha,
        "summary_sha256": _sha256(SUMMARY_PATH),
        "artifact_manifest_sha256": _sha256(MANIFEST_PATH),
        "artifact_count": manifest["artifact_count"],
        "promote_to_v13": False,
        "external_test_authorized": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
        "test_v12_used": False,
    }
    DECISION_RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
