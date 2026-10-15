"""One-time Route-A locked-test evaluation for the frozen anti-alias candidate.

The authorization and freeze-hash gate is evaluated before this module opens
the sealed source manifest or any MAT file. The runner refuses an existing
terminal receipt, so it cannot silently overwrite or repeat the sealed test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.engine.locked_test import sha256_file, validate_locked_test_access  # noqa: E402
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_c1_baseline import CLASS_NAMES, metrics  # noqa: E402

AUTHORIZATION_TOKEN = "ROUTE_A_LOCKED_TEST_ONCE_20260827"


class LockedSixDataset(Dataset):
    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None
        with h5py.File(str(self.path), "r") as handle:
            shape = tuple(handle["known_label"].shape)
            if shape[0] != 6:
                raise ValueError(f"expected six labels, got {shape}")
            self.labels = np.asarray(handle["known_label"][:, :], dtype=np.uint8).T
            self.sample_count = int(handle["iq_complex"].shape[1])
        if self.labels.shape != (self.sample_count, 6):
            raise ValueError(f"label/sample mismatch: {self.labels.shape}, {self.sample_count}")

    def __len__(self):
        return self.sample_count

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(str(self.path), "r")
        return self._handle

    def __getitem__(self, index):
        stored = np.asarray(self._file()["iq_complex"][:, int(index)])
        if stored.dtype.fields is None:
            raise TypeError("iq_complex must contain compound real/imag fields")
        signal = np.stack([
            np.asarray(stored["real"], dtype=np.float32),
            np.asarray(stored["imag"], dtype=np.float32),
        ], axis=0)
        rms = np.sqrt(np.mean(signal[0] ** 2 + signal[1] ** 2))
        if np.isfinite(rms) and rms > 0:
            signal = signal / np.sqrt(rms * rms + 1e-12)
        return torch.from_numpy(np.ascontiguousarray(signal)), torch.from_numpy(self.labels[int(index)].astype(np.float32))

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ece(targets: np.ndarray, logits: np.ndarray, bins: int = 15) -> float:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    truth = targets.reshape(-1).astype(np.float64)
    conf = probs.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(conf)); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            value += float(mask.sum()) / total * abs(float(conf[mask].mean()) - float(truth[mask].mean()))
    return value


def _extended_metrics(targets: np.ndarray, logits: np.ndarray) -> dict:
    result = metrics(targets, logits)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    preds = (probs >= 0.5).astype(np.uint8)
    result["label_cardinality_error"] = float(np.mean(np.abs(preds.astype(np.int64).sum(1) - targets.astype(np.int64).sum(1))))
    result["calibration_error_ece15"] = _ece(targets, logits)
    return result


def _load_models(entries: list[dict], model_name: str, device: torch.device) -> list[torch.nn.Module]:
    models = []
    for entry in entries:
        path = Path(entry["path"])
        actual = sha256_file(path)
        if actual.lower() != entry["sha256"].lower():
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        model = build_backbone(model_name, num_classes=6, width=16, dropout=0.2).to(device)
        model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True), strict=True)
        model.eval(); models.append(model)
    return models


def _evaluate_power(path: Path, candidate_models, baseline_models, device, batch_size: int):
    dataset = LockedSixDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    targets, candidate_logits, baseline_logits = [], [[] for _ in candidate_models], [[] for _ in baseline_models]
    try:
        with torch.inference_mode():
            for signal, target in loader:
                signal = signal.to(device, non_blocking=True)
                targets.append(target.numpy().astype(np.uint8))
                for i, model in enumerate(candidate_models):
                    candidate_logits[i].append(model(signal).cpu().numpy().astype(np.float32))
                for i, model in enumerate(baseline_models):
                    baseline_logits[i].append(model(signal).cpu().numpy().astype(np.float32))
    finally:
        dataset.close()
    y = np.concatenate(targets)
    c = [np.concatenate(parts) for parts in candidate_logits]
    b = [np.concatenate(parts) for parts in baseline_logits]
    return y, c, b


def _paired_stat(candidate: list[float], baseline: list[float]) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    try:
        statistic, pvalue = wilcoxon(delta, alternative="two-sided", zero_method="pratt")
        statistic, pvalue = float(statistic), float(pvalue)
    except ValueError:
        statistic, pvalue = 0.0, 1.0
    return {
        "candidate": [float(x) for x in candidate], "baseline": [float(x) for x in baseline],
        "paired_delta_mean": float(delta.mean()), "paired_delta_std": float(delta.std(ddof=1)),
        "direction_positive_count": int((delta > 0).sum()), "direction_negative_count": int((delta < 0).sum()),
        "wilcoxon_statistic": statistic, "wilcoxon_pvalue_two_sided": pvalue,
    }


def _write_markdown(path: Path, result: dict) -> None:
    """Write a compact, deterministic human-readable report after raw JSON."""
    pooled = result["pooled"]
    c = pooled["candidate_ensemble"]
    b = pooled["baseline_ensemble"]
    lines = [
        "# Route-A one-time locked-test result",
        "",
        "This report is generated from `raw_locked_test_result.json`; the raw JSON is authoritative.",
        "",
        "- classes: " + ", ".join(result["classes"]),
        "- powers (dB): " + ", ".join(str(x) for x in result["powers_db"]),
        "- threshold: " + str(result["threshold"]),
        "- aggregation: " + result["aggregation"],
        "",
        "## Pooled ensemble",
        "",
        "| metric | candidate F2_anti_alias | baseline F2_full | delta (candidate-baseline) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("macro_aupr", "macro_auroc", "macro_f1", "hamming_loss", "subset_accuracy", "combination_exact_accuracy", "label_cardinality_error", "calibration_error_ece15"):
        lines.append(f"| {name} | {c[name]:.6f} | {b[name]:.6f} | {c[name] - b[name]:+.6f} |")
    lines.extend(["", "## Per-power ensemble", "", "| power dB | candidate macro-AUPR | baseline macro-AUPR | candidate combo exact | baseline combo exact |", "|---:|---:|---:|---:|---:|"])
    for power in sorted(result["per_power"], key=lambda x: float(x)):
        row = result["per_power"][power]
        cc, bb = row["candidate_ensemble"], row["baseline_ensemble"]
        lines.append(f"| {power} | {cc['macro_aupr']:.6f} | {bb['macro_aupr']:.6f} | {cc['combination_exact_accuracy']:.6f} | {bb['combination_exact_accuracy']:.6f} |")
    lines.extend(["", "## Paired checkpoint statistics", "", "| metric | mean delta | Wilcoxon p (two-sided) | positive / negative |", "|---|---:|---:|---:|"])
    for name, row in pooled["paired_checkpoint_statistics"].items():
        lines.append(f"| {name} | {row['paired_delta_mean']:+.6f} | {row['wilcoxon_pvalue_two_sided']:.6g} | {row['direction_positive_count']} / {row['direction_negative_count']} |")
    lines.extend(["", "Post-read policy: no model, threshold, checkpoint, preprocessing, split, or metric-definition changes are permitted.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    if args.authorization_token != AUTHORIZATION_TOKEN:
        raise PermissionError("explicit one-time authorization token is required")
    freeze_path = validate_locked_test_access(True, args.freeze_contract, args.expected_freeze_sha256)
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "frozen_waiting_explicit_authorization":
        raise RuntimeError("execution freeze status drift")
    runner_path = Path(freeze["runner"])
    if runner_path.resolve() != Path(__file__).resolve():
        raise RuntimeError("runner path drift")
    expected_runner_sha256 = str(freeze.get("runner_sha256", ""))
    if not expected_runner_sha256 or sha256_file(runner_path).lower() != expected_runner_sha256.lower():
        raise RuntimeError("runner hash mismatch")
    output = Path(freeze["output_root"])
    terminal = output / "terminal_receipt.json"
    if terminal.exists():
        raise RuntimeError(f"one-time receipt already exists; refusing repeat: {terminal}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"one-time output root is not empty; refusing overwrite: {output}")
    source_manifest_path = Path(freeze["sealed_source_manifest"])
    candidate_manifest_path = Path(freeze["candidate_manifest"])
    for label, path, expected in (
        ("sealed source manifest", source_manifest_path, freeze.get("sealed_source_manifest_sha256", "")),
        ("candidate manifest", candidate_manifest_path, freeze.get("candidate_manifest_sha256", "")),
    ):
        if not expected or sha256_file(path).lower() != str(expected).lower():
            raise RuntimeError(f"{label} hash mismatch")
    source = _read_json(source_manifest_path)
    candidate_manifest = _read_json(candidate_manifest_path)
    if source["status"] != "sealed_waiting_explicit_authorization":
        raise RuntimeError("sealed source status drift")
    if candidate_manifest["threshold"] != 0.5:
        raise RuntimeError("threshold drift")
    if candidate_manifest["selected_classes"] != list(CLASS_NAMES):
        raise RuntimeError("class-order drift")
    for entry in source["files"]:
        if sha256_file(entry["path"]).lower() != entry["sha256"].lower():
            raise RuntimeError(f"sealed MAT hash mismatch: {entry['path']}")
    candidate_entries = freeze["candidate_checkpoints"]
    baseline_entries = freeze["baseline_checkpoints"]
    if len(candidate_entries) != 10 or len(baseline_entries) != 10:
        raise RuntimeError("exactly ten paired checkpoints are required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the preregistered one-time run requires CUDA")
    # PyTorch 2.11 on this workstation rejects a torch.device object here
    # until the selected device has been made current; the no-argument form
    # preserves the same measurement contract after explicit initialization.
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    candidate_models = _load_models(candidate_entries, "dfms_iq_anti_alias", device)
    baseline_models = _load_models(baseline_entries, "dfms_iq", device)
    started = time.time(); power_results = {}; pooled = {"targets": [], "candidate": [[] for _ in range(10)], "baseline": [[] for _ in range(10)]}
    for entry in sorted(source["files"], key=lambda row: row["power_db"]):
        y, c_logits, b_logits = _evaluate_power(Path(entry["path"]), candidate_models, baseline_models, device, args.batch_size)
        c_ensemble = np.mean(np.stack(c_logits), axis=0); b_ensemble = np.mean(np.stack(b_logits), axis=0)
        power_results[str(entry["power_db"])] = {
            "sample_count": int(len(y)), "candidate_ensemble": _extended_metrics(y, c_ensemble),
            "baseline_ensemble": _extended_metrics(y, b_ensemble),
            "candidate_per_checkpoint": [_extended_metrics(y, z) for z in c_logits],
            "baseline_per_checkpoint": [_extended_metrics(y, z) for z in b_logits],
        }
        pooled["targets"].append(y)
        for i in range(10): pooled["candidate"][i].append(c_logits[i]); pooled["baseline"][i].append(b_logits[i])
    y_all = np.concatenate(pooled["targets"])
    c_all = [np.concatenate(parts) for parts in pooled["candidate"]]
    b_all = [np.concatenate(parts) for parts in pooled["baseline"]]
    c_ensemble = np.mean(np.stack(c_all), axis=0); b_ensemble = np.mean(np.stack(b_all), axis=0)
    c_metrics = [_extended_metrics(y_all, z) for z in c_all]; b_metrics = [_extended_metrics(y_all, z) for z in b_all]
    paired = {name: _paired_stat([m[name] for m in c_metrics], [m[name] for m in b_metrics]) for name in ("macro_aupr", "macro_f1", "combination_exact_accuracy", "hamming_loss")}
    result = {
        "schema": "route-a-locked-test-once-raw-result-v1", "status": "completed_one_time_locked_test",
        "locked_test_used": True, "read_count": 1, "classes": list(CLASS_NAMES), "threshold": 0.5,
        "candidate": "F2_anti_alias/dfms_iq_anti_alias", "baseline": "F2_full/dfms_iq",
        "aggregation": "uniform arithmetic mean of ten logits", "powers_db": source["powers_db"],
        "per_power": power_results,
        "pooled": {"candidate_ensemble": _extended_metrics(y_all, c_ensemble), "baseline_ensemble": _extended_metrics(y_all, b_ensemble), "paired_checkpoint_statistics": paired},
        "cost": {"elapsed_seconds": float(time.time() - started), "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "candidate_parameters_each": int(sum(p.numel() for p in candidate_models[0].parameters())), "baseline_parameters_each": int(sum(p.numel() for p in baseline_models[0].parameters()))},
        "integrity": {"freeze_contract": str(freeze_path), "freeze_contract_sha256": args.expected_freeze_sha256, "sealed_source_manifest": str(source_manifest_path), "candidate_manifest": str(candidate_manifest_path)},
        "post_read_policy": "No model, threshold, checkpoint, preprocessing, split, or metric-definition changes are permitted."
    }
    raw_path = output / "raw_locked_test_result.json"
    _write_json(raw_path, result)
    report_path = output / "locked_test_report.md"
    _write_markdown(report_path, result)
    terminal_payload = {"status": "completed_one_time_locked_test", "read_count": 1, "locked_test_used": True, "raw_result": str(raw_path.resolve()), "raw_result_sha256": sha256_file(raw_path), "report": str(report_path.resolve()), "report_sha256": sha256_file(report_path), "runner_sha256": sha256_file(Path(__file__)), "completed_at_unix": time.time(), "post_read_changes_prohibited": True}
    _write_json(terminal, terminal_payload)
    print(json.dumps({"status": terminal_payload["status"], "raw_result": str(raw_path), "terminal_receipt": str(terminal)}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--freeze-contract", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
