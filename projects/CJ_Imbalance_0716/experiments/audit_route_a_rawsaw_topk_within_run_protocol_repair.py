"""Read-only integrity audit for the completed equal-budget RawSaw top-k study."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPKS = (4, 8, 16)

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _build(source_root, smoke=False):
    source_path = source_root / "route_a_rawsaw_topk_4_8_16_ablation_summary.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("smoke") or source.get("topks") != list(TOPKS):
        raise ValueError("source is not the completed 4/8/16 top-k study")
    seeds = source["seeds"][:1] if smoke else source["seeds"]
    rows = []
    for seed in seeds:
        path = source_root / f"seed_{seed}" / "seed_summary.json"
        rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
    checks = {
        "five_seed_source": len(source["seeds"]) == 5,
        "source_not_smoke": source.get("smoke") is False,
        "same_data_sha": len({row["data_sha256"] for _, row in rows}) == 1 and source["data_sha256"] == rows[0][1]["data_sha256"],
        "same_trainable_parameter_count_within_seed": all(len({row["training"][f"RawSaw_topk{k}"]["trainable_parameters"] for k in TOPKS}) == 1 for _, row in rows),
        "non_saw_probabilities_exact": all(max(item["non_saw_probability_max_abs"] for item in row["non_saw_safety"].values()) == 0. for _, row in rows),
        "non_saw_predictions_exact": all(sum(item["non_saw_prediction_mismatch"] for item in row["non_saw_safety"].values()) == 0 for _, row in rows),
        "test_boundary_safe": source.get("locked_test_used") is False and source.get("test_v2_used") is False and source.get("test_v3_used") is False and all(not row["locked_test_used"] and not row["test_v2_used"] and not row["test_v3_used"] for _, row in rows),
    }
    drift = {str(row["seed"]): float(row["topk16_reference_state_max_abs"]) for _, row in rows}
    return {"status": "completed", "experiment": "RouteA_RawSawTopKWithinRunProtocolRepair", "smoke": bool(smoke), "source_summary": str(source_path), "source_summary_sha256": _sha256(source_path), "seeds": seeds, "data_sha256": source["data_sha256"], "source_comparisons_vs_topk16": source["comparisons_vs_topk16"], "historical_topk16_state_drift_max_abs": drift, "historical_bitwise_reproduction": all(v <= 1e-6 for v in drift.values()), "within_run_integrity": {"checks": checks, "passed": bool(all(checks.values()))}, "scope": "read_only_within_run_pooling_ablation_no_candidate_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_rawsaw_topk_4_8_16_ablation")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_rawsaw_topk_within_run_protocol_repair")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    payload = _build(args.source_root, args.smoke)
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "route_a_rawsaw_topk_within_run_protocol_repair_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
