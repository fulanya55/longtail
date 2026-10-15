from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RUN = ROOT / "outputs" / "route_b_r1_full_development" / "r1_20260826_122730"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def nums(rows, key):
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def mean(rows, key):
    values = nums(rows, key)
    return statistics.fmean(values) if values else float("nan")


def fmt(value, digits=5):
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def main() -> None:
    aggregate = read_json(DOCS / "results" / "route_b_r1_aggregate.json")
    r3_receipt = read_json(DOCS / "receipts" / "route_b_r3_result_receipt.json")
    stop_receipt = read_json(DOCS / "receipts" / "route_b_abort_receipt.json")
    seed_rows = read_csv(RUN / "per_seed_summary.csv")
    power_rows = read_csv(DOCS / "results" / "route_b_paired_power_consistency.csv")
    visibility_rows = read_csv(DOCS / "results" / "route_b_visibility_audit.csv")
    parameter_rows = read_csv(DOCS / "results" / "route_b_per_parameter_metrics.csv")

    baseline_path = DOCS / "results" / "route_b_r1_baseline.csv"
    with baseline_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(seed_rows[0].keys()) + ["claim_scope"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow({**row, "claim_scope": "development_only_no_locked_test"})

    by_variant_power = {}
    for variant in ("B1", "B2", "B3", "B4"):
        by_variant_power[variant] = {}
        rows_v = [r for r in power_rows if r["variant"] == variant]
        for level in sorted({int(float(r["jnr_dB"])) for r in rows_v}):
            rows = [r for r in rows_v if int(float(r["jnr_dB"])) == level]
            by_variant_power[variant][level] = {
                "aupr": mean(rows, "macro_aupr"),
                "f1": mean(rows, "macro_f1"),
                "nmae": mean(rows, "parameter_macro_nmae"),
            }

    visibility_summary = {}
    for variant in ("B1", "B2", "B3", "B4"):
        rows = [r for r in visibility_rows if r["variant"] == variant and r["power_role"] == "nominal_train_val"]
        visibility_summary[variant] = {
            "oracle_coverage": mean(rows, "oracle_coverage"),
            "predicted_gate_rate": mean(rows, "predicted_gate_rate"),
            "safe_stop_rate": mean(rows, "safe_stop_rate_on_negative"),
        }

    parameter_summary = []
    for variant in ("B1", "B2", "B3", "B4"):
        rows = [r for r in parameter_rows if r["variant"] == variant and r["power_role"] == "nominal_train_val"]
        for class_name in sorted({r["class_name"] for r in rows}):
            class_rows = [r for r in rows if r["class_name"] == class_name]
            parameter_summary.append(
                f"| {variant} | {class_name} | {fmt(mean(class_rows, 'oracle_nmae'))} | {fmt(mean(class_rows, 'end_to_end_nmae'))} |"
            )

    lines = [
        "# Route B final evidence report (development-only, gate-audited)",
        "",
        "## 1. Research question",
        "",
        "Test whether conditional parameter heads improve masked estimation for the six fixed classes while preserving the recognition task under the fixed [N,2,3000] I/Q contract.",
        "",
        "## 2. Data roles, identity split and hashes",
        "",
        "R0 verified the active six-class train MAT, finite labels/targets and masks, and unique identity grouping. The development split is seed 20260826 with 22,680 train and 2,520 validation identities. The historic 0716 locked test was not opened. Nine power views were treated as paired views sharing receiver_noise_seed, never as independent samples. See the manifest receipt for current code/config/data-role hashes.",
        "",
        "## 3. Models",
        "",
        "B0 is classification-only; B1 uses a global joint parameter head; B2 is class-conditioned; B3 adds stop-gradient class probabilities; B4 warms the parameter loss for two epochs. Physical-FiLM (26 features) and PSD spectral-FiLM (19 features) are implemented and smoke-verified only; they are not performance-promoted.",
        "",
        "## 4. Training protocol",
        "",
        "All R1 variants used the same split, AdamW (lr 3e-4, weight decay 1e-4), masked SmoothL1 parameter loss, parameter weight 0.1, threshold 0.5, batch size 64, width 16, head dimension 256, hidden dimension 64, 12 epochs maximum, patience 4, and seeds 2041--2045. Aggregation uses deterministic percentile bootstrap 95% CIs over development seeds.",
        "",
        "## 5. Validity and masking",
        "",
        "A parameter contributes only when label=1, parameter_mask=1 and target is finite. Inactive NaNs were not zero-filled; normalization statistics were fitted only on valid training targets.",
        "",
        "## 6. R1 classification and parameter results",
        "",
        "| Variant | Macro-AUPR mean | Macro-F1 mean | Parameter Macro-NMAE mean | Gate status |",
        "|---|---:|---:|---:|---|",
    ]
    for variant in ("B0", "B1", "B2", "B3", "B4"):
        agg = aggregate["aggregate"][variant]
        status = "reference" if variant in ("B0", "B1") else ("stopped: parameter gate failed")
        lines.append(f"| {variant} | {fmt(agg['macro_aupr']['mean'])} | {fmt(agg['macro_f1']['mean'])} | {fmt(agg['parameter_macro_nmae']['mean'])} | {status} |")
    lines += [
        "",
        "B0 Macro-AUPR=0.98844 and Macro-F1=0.94593. B1 parameter Macro-NMAE=0.57680. B2/B3/B4 preserved the preregistered classification safety limits, but their parameter relative improvements versus B1 were -5.49%, -8.57% and -8.78%, respectively. The 5% parameter-improvement hard gate therefore failed for all three conditional branches. The stop receipt records that no threshold retuning, resampling, model enlargement or test feedback was used.",
        "",
        "## 7. Per-class parameter evidence",
        "",
        "| Variant | Class | Oracle NMAE | End-to-end NMAE |",
        "|---|---|---:|---:|",
        *parameter_summary,
        "",
        "## 8. Visibility and low-power evidence",
        "",
        "Nominal validation visibility summaries (corrected denominator: two parameters per class per identity) are:",
        "",
        "| Variant | Oracle coverage | Predicted gate rate | Safe-stop rate on negatives |",
        "|---|---:|---:|---:|",
    ]
    for variant, summary in visibility_summary.items():
        lines.append(f"| {variant} | {fmt(summary['oracle_coverage'])} | {fmt(summary['predicted_gate_rate'])} | {fmt(summary['safe_stop_rate'])} |")
    lines += [
        "",
        "These are development validation quantities only, not an open-world or locked-test claim.",
        "",
        "## 9. Paired-power consistency",
        "",
        "The same validation identities were evaluated over JNR 10--50 dB train views. Mean values below are per variant and JNR; they are descriptive paired-view evidence, not independent-sample confidence intervals.",
        "",
        "| Variant | JNR dB | Macro-AUPR | Macro-F1 | Parameter Macro-NMAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("B1", "B2", "B3", "B4"):
        for level, values in by_variant_power[variant].items():
            lines.append(f"| {variant} | {level} | {fmt(values['aupr'])} | {fmt(values['f1'])} | {fmt(values['nmae'])} |")
    lines += [
        "",
        "## 10. Multi-seed and resource evidence",
        "",
        "All five variants have five completed development seeds. Resource measurements are recorded separately in route_b_resource_metrics.csv and are approximate inference benchmarks, not dataset throughput claims.",
        "",
        "## 11. Failures and stop branches",
        "",
        "R1 B2, B3 and B4 are stopped because the parameter hard gate failed. R4's ten strict ablations were preregistered and preflight-smoke-verified but not executed because their conditional-head prerequisites were not met. R2 physical/spectral branches remain proposal/smoke evidence only. No locked test or test-v2 result is claimed.",
        "",
        "## 12. Literature correspondence",
        "",
        "The literature-to-experiment plan maps communication/radar/emitter/jamming parameter-extraction findings to the shared I/Q backbone, spectral/physical auxiliary hypotheses, conditional masking and paired-power checks. Each mapping remains a hypothesis until its matched experiment passes the preregistered gate.",
        "",
        "## 13. Claimable conclusions",
        "",
        "- R0 data, identity split, mask semantics, code-path and smoke readiness are verified.",
        "- B1 is the only retained joint baseline from the completed development gate; B2/B3/B4 are not promoted.",
        "- Train-only paired-power and visibility tables are complete with locked-test usage false.",
        "",
        "## 14. Not claimable",
        "",
        "No final-model superiority, conditional-head gain, low-power robustness, open-world generalization, independent test-v2 performance, or R4 ablation conclusion is claimable.",
        "",
        "## 15. Next minimum experiment",
        "",
        "Design a fresh preregistered parameter-head hypothesis or revise the objective, then rerun identity-isolated development-v2 with the same hard-gate discipline before any independent test-v2 evaluation.",
    ]
    (DOCS / "reports" / "route_b_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result_receipt = {
        "stage": "Route_B_final",
        "status": "verified_development_with_stopped_branches",
        "r0_status": "verified",
        "r1_status": "verified_development_parameter_gate_failed",
        "r1_gate_pass": False,
        "r1_stopped_variants": stop_receipt.get("failed_branches", ["B2", "B3", "B4"]),
        "r2_status": "verified_smoke_only_not_performance_validated",
        "r3_status": r3_receipt["status"],
        "r4_status": "not_run_prerequisite_failed",
        "locked_test_used": False,
        "test_mat_opened": False,
        "development_seed_count": 5,
        "r3_rows": r3_receipt["rows"],
        "claim_scope": "development_only_no_locked_test_or_test_v2_claim",
    }
    (DOCS / "receipts" / "route_b_result_receipt.json").write_text(json.dumps(result_receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    index = {
        "goal": "Route B conditional parameter estimation",
        "r0": str(DOCS / "receipts" / "route_b_r0_result_receipt.json"),
        "r0_report": str(DOCS / "reports" / "route_b_r0_readiness.md"),
        "r1": str(RUN / "per_seed_summary.csv"),
        "r1_status": "verified_development_parameter_gate_failed",
        "r1_runner_smoke": str(DOCS / "receipts" / "route_b_r1_runner_smoke_receipt.json"),
        "r1_aggregate": str(DOCS / "receipts" / "route_b_r1_aggregate_receipt.json"),
        "r1_aggregate_status": "verified_development_hard_gate_failed",
        "r1_abort_receipt": str(DOCS / "receipts" / "route_b_abort_receipt.json"),
        "r2_r3_preflight": str(DOCS / "receipts" / "route_b_r2_r3_preflight_receipt.json"),
        "r2_status": "verified_smoke_only_not_performance_validated",
        "r3_result_receipt": str(DOCS / "receipts" / "route_b_r3_result_receipt.json"),
        "r3_status": "verified_development_train_only",
        "r4_preflight": str(DOCS / "receipts" / "route_b_r4_preflight_receipt.json"),
        "r4_stop_report": str(DOCS / "reports" / "route_b_r4_stop.md"),
        "r4_status": "not_run_prerequisite_failed",
        "literature_plan": str(DOCS / "reports" / "route_b_literature_to_experiment_plan.md"),
        "final_report": str(DOCS / "reports" / "route_b_final_report.md"),
        "final_result_receipt": str(DOCS / "receipts" / "route_b_result_receipt.json"),
        "r3_evaluator": str(ROOT / "scripts" / "route_b_r3_evaluate.py"),
        "r4_preregister": str(ROOT / "configs" / "route_b_r4_preregister.json"),
        "manifest": str(DOCS / "receipts" / "route_b_manifest_receipt.json"),
        "resource_benchmark": str(DOCS / "receipts" / "route_b_resource_receipt.json"),
        "locked_test_used": False,
    }
    (DOCS / "receipts" / "route_b_final_evidence_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
