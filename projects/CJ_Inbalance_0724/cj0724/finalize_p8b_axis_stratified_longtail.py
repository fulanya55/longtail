"""Write the P8b development receipt from completed, immutable run artifacts only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.dual_axis import audit_power_alignment, build_identity_roles, discover_power_train_files, read_identity_metadata, roles_receipt
from cj0724.p8a_multi_axis_longtail_atlas import build_manifest, sha256_file
from cj0724.p8b_axis_stratified_longtail import POWERS, VFT, _bootstrap_vft_aupr_delta, _gates, _json_safe, _metric_summary


def _recover_run(output: Path, name: str, seed: int) -> dict:
    checkpoint = output / f"{name}__seed{seed}.pt"
    prediction = output / f"{name}__seed{seed}_development.npz"
    if not checkpoint.is_file() or not prediction.is_file():
        raise FileNotFoundError(f"Missing completed artifact for {name}, seed {seed}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    thresholds = np.asarray(state["thresholds"], dtype=np.float64)
    powers, raw = {}, {}
    with np.load(prediction, allow_pickle=False) as archive:
        for power in POWERS:
            key = str(power)
            labels = np.asarray(archive[f"{key}_labels"], dtype=np.uint8)
            probability = np.asarray(archive[f"{key}_probability"], dtype=np.float64)
            indices = np.asarray(archive[f"{key}_indices"], dtype=np.int64)
            weak = np.zeros_like(labels, dtype=bool)
            weak[:, VFT] = (power < 30) & (labels[:, VFT] == 1) & labels[:, :3].any(axis=1)
            powers[key] = _metric_summary(labels, probability, thresholds, weak)
            raw[key] = {"labels": labels, "probability": probability, "indices": indices}
    return {
        "name": name, "seed": int(seed), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": int(state["best_epoch"]),
        "thresholds": thresholds.tolist(), "history": state["history"], "powers": powers, "raw_10": raw["10"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_root / "result.json").exists():
        raise FileExistsError("P8b result receipt already exists; refuse to overwrite it")
    cfg = json.loads(args.preregister.read_text(encoding="utf-8"))
    freeze = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    if sha256_file(args.preregister) != freeze["preregister_sha256"]:
        raise RuntimeError("P8b preregistration hash mismatch")
    files = discover_power_train_files(args.data_root)
    if any("test" in str(path).lower() for path in files.values()):
        raise RuntimeError("P8b finalization permits only train paths")
    p8a = json.loads(Path(cfg["parent_protocol"]["p8a_result"]).read_text(encoding="utf-8"))
    fresh_ids, labels = read_identity_metadata(files[30])
    old_ids, _ = read_identity_metadata(Path(p8a["train_only_power_alignment"]["levels"]["30"]["path"]))
    if np.intersect1d(fresh_ids, old_ids).size:
        raise RuntimeError("Fresh P8b identities overlap P8a root")
    roles = build_identity_roles(labels, int(cfg["roles"]["split_seed"]))
    manifest = build_manifest(labels, roles["optimization"], "mixed_lt", cfg["roles"]["training_manifest"]["label_targets"], "combo_lt", 1310000003, True)
    runs = []
    for seed in cfg["models"]["model_seeds_development"]:
        c0 = _recover_run(args.output_root, "C0_random_BCE", int(seed))
        c1 = _recover_run(args.output_root, "C1_axis_stratified_BCE", int(seed))
        a, b = c0["powers"]["10"], c1["powers"]["10"]
        delta = {
            "vft_10db_aupr_delta": b["per_label_aupr"][VFT] - a["per_label_aupr"][VFT],
            "macro_10db_aupr_delta": b["macro_aupr"] - a["macro_aupr"],
            "hamming_10db_delta": b["hamming_loss"] - a["hamming_loss"],
            "hard_fpr_10db_delta": b["vft_hard_negative_fpr"] - a["vft_hard_negative_fpr"],
            "vft_separation_10db_delta": b["vft_separation"] - a["vft_separation"],
            "worst_power_vft_fpr_delta": max(c1["powers"][str(power)]["per_label_fpr"][VFT] - c0["powers"][str(power)]["per_label_fpr"][VFT] for power in POWERS),
            "non_vft_fpr_delta": [b["per_label_fpr"][index] - a["per_label_fpr"][index] for index in range(6) if index != VFT],
        }
        bootstrap = _bootstrap_vft_aupr_delta(c0["raw_10"]["labels"], c0["raw_10"]["probability"], c1["raw_10"]["probability"], int(seed) + 500)
        c0.pop("raw_10"); c1.pop("raw_10")
        runs.append({"seed": int(seed), "C0": c0, "C1": c1, "delta": delta, "bootstrap_vft_10db_aupr": bootstrap})
    gates = _gates(runs)
    result = {
        "experiment": cfg["experiment"], "preregister_sha256": sha256_file(args.preregister),
        "method_freeze_sha256": sha256_file(args.method_freeze), "frozen_training_runner_sha256": freeze["runner_sha256"],
        "finalizer_sha256": sha256_file(Path(__file__)), "train_only_alignment": audit_power_alignment(files),
        "roles": roles_receipt(roles, labels, int(cfg["roles"]["split_seed"])), "manifest": manifest,
        "runs": runs, "gates": gates, "development_passed": bool(all(gates.values())),
        "verdict": "promote_to_two_model_seed_extension" if all(gates.values()) else "stop_no_confirmation_data",
        "test_generated": False, "test_opened": False, "locked_test_used": False,
        "postprocess_note": "Recovered solely from six completed checkpoints and six saved development predictions after the original JSON serialization failure; no model was retrained.",
    }
    (args.output_root / "result.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"development_passed": result["development_passed"], "gates": gates, "verdict": result["verdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
