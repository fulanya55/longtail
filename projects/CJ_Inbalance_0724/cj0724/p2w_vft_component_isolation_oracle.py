"""P2w oracle sufficiency screen: raw composite versus exact isolated VFT component."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from cj0724.dual_axis import read_identity_metadata
from cj0724.p8a_multi_axis_longtail_atlas import sha256_file
from cj0724.vft_doppler_aligned_expert import estimate_aligned_doppler

VFT, DOPPLER = 4, 49
POWER_RE = re.compile(r"jnr_(\d+)dB$")


def power_files(root: Path) -> dict[int, Path]:
    files = {}
    for path in root.glob("level_*__jnr_*dB/closedset_6class_train.mat"):
        match = POWER_RE.search(path.parent.name)
        if match:
            files[int(match.group(1))] = path
    if set(files) != {10, 30, 50}:
        raise RuntimeError(f"P2w requires exactly 10/30/50 dB train files: {root}")
    return files


def _to_iq(raw: np.ndarray) -> torch.Tensor:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6000:
        raise ValueError(f"P2w expects sample-first [N,6000] legacy I/Q, got {values.shape}")
    return torch.from_numpy(values.reshape(-1, 2, 3000))


def _estimate(iq: np.ndarray, kwargs: dict, device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, len(iq), 32):
        batch = _to_iq(iq[start:start + 32]).to(device)
        output.append(estimate_aligned_doppler(batch, **kwargs)["prediction_hz"].cpu().numpy())
    return np.concatenate(output)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    finite = np.isfinite(prediction)
    error = np.abs(prediction - target)
    correlation = float(np.corrcoef(target, prediction)[0, 1]) if len(target) > 1 and np.std(target) > 0 and np.std(prediction) > 0 else float("nan")
    return {"finite_rate": float(finite.mean()), "mae_hz": float(error.mean()), "pearson": correlation, "absolute_error_hz": error}


def evaluate_file(path: Path, device: torch.device, correct: dict) -> dict:
    if "test" in str(path).lower():
        raise RuntimeError("P2w refuses test path")
    oracle_path = path.with_name("vft_oracle_component.mat")
    if not oracle_path.exists():
        raise RuntimeError(f"Missing oracle component file: {oracle_path}")
    with h5py.File(path, "r") as handle, h5py.File(oracle_path, "r") as oracle:
        labels = np.asarray(handle["label_selected"], dtype=np.uint8).T
        positive = np.flatnonzero(labels[:, VFT] == 1)
        indices = np.asarray(oracle["vft_oracle_indices"], dtype=np.int64).reshape(-1) - 1
        if not np.array_equal(indices, positive):
            raise RuntimeError("Oracle indices do not exactly equal VFT-positive identities")
        target = np.asarray(handle["parameter_values"][DOPPLER, positive], dtype=np.float64)
        mask = np.asarray(handle["parameter_mask"][DOPPLER, positive], dtype=np.uint8).astype(bool)
        if not mask.all() or not np.isfinite(target).all():
            raise RuntimeError("P2w VFT Doppler target contract violated")
        raw = np.asarray(handle["data_legacy"][:, positive], dtype=np.float32).T
        clean = np.asarray(oracle["vft_oracle_iq"], dtype=np.float32)
    c0 = _estimate(raw, correct, device)
    c1 = _estimate(clean, correct, device)
    wrong = dict(correct); wrong["chirp_rate_hz_per_s"] *= 0.8
    no_delay = dict(correct); no_delay["search_delay"] = False; no_delay["delay_min_samples"] = 0; no_delay["delay_max_samples"] = 0
    c2, c3 = _estimate(clean, wrong, device), _estimate(clean, no_delay, device)
    return {"path": str(path), "positive_count": int(len(positive)), "C0_raw": _metrics(target, c0), "C1_oracle": _metrics(target, c1), "C2_wrong_reference": _metrics(target, c2), "C3_no_delay": _metrics(target, c3)}


def gate(rows: list[dict]) -> dict:
    c1 = [row["C1_oracle"] for row in rows]; c0 = [row["C0_raw"] for row in rows]
    per = [item["mae_hz"] / base["mae_hz"] for item, base in zip(c1, c0)]
    by_power = {power: [row for row in rows if row["power_db"] == power] for power in (10, 30, 50)}
    controls = {}
    for power, subset in by_power.items():
        oracle_mae = np.mean([row["C1_oracle"]["mae_hz"] for row in subset])
        controls[str(power)] = {"wrong_reference_ratio": oracle_mae / np.mean([row["C2_wrong_reference"]["mae_hz"] for row in subset]), "no_delay_ratio": oracle_mae / np.mean([row["C3_no_delay"]["mae_hz"] for row in subset])}
    return {
        "C1_finite_rate_each_seed_power": all(item["finite_rate"] == 1.0 for item in c1),
        "C1_pearson_each_seed_power_min": all(item["pearson"] >= 0.7 for item in c1),
        "C1_mae_each_seed_power_vs_C0_ratio_max": all(value <= 0.8 for value in per),
        "C1_10dB_mae_each_seed_vs_C0_ratio_max": all(value <= 0.5 for value, row in zip(per, rows) if row["power_db"] == 10),
        "C1_mean_mae_each_power_vs_C2_ratio_max": all(value["wrong_reference_ratio"] <= 0.8 for value in controls.values()),
        "C1_mean_mae_each_power_vs_C3_ratio_max": all(value["no_delay_ratio"] <= 0.8 for value in controls.values()),
        "mae_ratio_by_seed_power": per, "control_ratios_by_power": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--data-roots", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.preregister.read_text(encoding="utf-8")); freeze = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    if sha256_file(args.preregister) != freeze["preregister_sha256"] or sha256_file(Path(__file__)) != freeze["runner_sha256"]:
        raise RuntimeError("P2w method freeze mismatch")
    if args.output.exists(): raise FileExistsError(args.output)
    identities = []; rows = []; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = dict(cfg["frozen_estimator"]["correct_reference"])
    for seed, root in zip(cfg["fresh_data"]["identity_seeds"], args.data_roots):
        files = power_files(root); ids, _ = read_identity_metadata(files[30]); identities.append(ids)
        if len(np.unique(ids)) != 630: raise RuntimeError("P2w identity count contract violated")
        for power, path in files.items():
            row = evaluate_file(path, device, correct); row.update({"seed": int(seed), "power_db": int(power)}); rows.append(row)
    if any(np.intersect1d(a, b).size for i, a in enumerate(identities) for b in identities[i + 1:]): raise RuntimeError("P2w seed pools overlap")
    gates = gate(rows); passed = bool(all(value for key, value in gates.items() if not key.endswith("by_seed_power") and not key.endswith("by_power")))
    for row in rows:
        for name in ("C0_raw", "C1_oracle", "C2_wrong_reference", "C3_no_delay"): row[name].pop("absolute_error_hz")
    output = {"experiment": cfg["experiment"], "preregister_sha256": sha256_file(args.preregister), "method_freeze_sha256": sha256_file(args.method_freeze), "rows": rows, "gates": gates, "oracle_sufficient": passed, "verdict": "permit_new_pool_separator_preregistration" if passed else "stop_separator_and_router", "test_generated": False, "test_opened": False, "locked_test_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"oracle_sufficient": passed, "verdict": output["verdict"]}, ensure_ascii=False))


if __name__ == "__main__": main()
