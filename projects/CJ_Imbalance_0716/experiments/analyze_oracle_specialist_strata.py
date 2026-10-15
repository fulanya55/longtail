"""Stratify oracle specialist errors by compound cardinality and component JNR."""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_shared import _data_path, _write_json, load_config  # noqa: E402


TARGETS = {
    "MFT_ACF_ORD": {"jnr_column": 27},
    "VFT_RAW": {"jnr_column": 48},
    "VFT_STFT": {"jnr_column": 48},
    "VFT_IFREQ": {"jnr_column": 48},
}


def _metrics(prediction, target, constant):
    error = np.abs(prediction - target)
    constant_error = np.abs(constant - target)
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if len(target) > 2 and prediction.std() > 0 and target.std() > 0
        else 0.0
    )
    return {
        "count": int(len(target)),
        "mae": float(error.mean()),
        "constant_mae": float(constant_error.mean()),
        "relative_reduction_vs_constant": float(
            1.0 - error.mean() / max(constant_error.mean(), 1.0e-12)
        ),
        "correlation": correlation,
    }


def analyze(config_path, output_root, output_path):
    config = load_config(Path(config_path))
    data_path = _data_path(config)
    labels = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    cardinality = labels.sum(axis=1)
    result = {"variants": {}}
    for variant, spec in TARGETS.items():
        records = []
        for seed in (2041, 2042, 2043):
            run_dir = Path(output_root) / variant / "seed_{}".format(seed)
            prediction_file = np.load(run_dir / "predictions.npz")
            with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            sample_ids = prediction_file["sample_ids"].astype(np.int64)
            records.append(
                {
                    "prediction": prediction_file["predictions"].astype(np.float64),
                    "target": prediction_file["targets"].astype(np.float64),
                    "constant": np.full(
                        len(sample_ids), float(metrics["training_median"]), dtype=np.float64
                    ),
                    "cardinality": cardinality[sample_ids],
                    "jnr": parameter_values[sample_ids, spec["jnr_column"]],
                }
            )
        joined = {
            key: np.concatenate([record[key] for record in records])
            for key in records[0]
        }
        by_cardinality = {}
        for value in range(1, 7):
            selected = joined["cardinality"] == value
            if selected.any():
                by_cardinality[str(value)] = _metrics(
                    joined["prediction"][selected],
                    joined["target"][selected],
                    joined["constant"][selected],
                )
        by_jnr = {}
        for lower in (10, 15, 20, 25):
            upper = lower + 5
            selected = np.logical_and(joined["jnr"] >= lower, joined["jnr"] < upper)
            if selected.any():
                by_jnr["{}-{}".format(lower, upper)] = _metrics(
                    joined["prediction"][selected],
                    joined["target"][selected],
                    joined["constant"][selected],
                )
        result["variants"][variant] = {
            "overall": _metrics(
                joined["prediction"], joined["target"], joined["constant"]
            ),
            "by_compound_cardinality": by_cardinality,
            "by_component_jnr_db": by_jnr,
        }
    with h5py.File(str(data_path), "r") as handle:
        fs = float(np.asarray(handle["config/radar/fs_Hz"]).reshape(-1)[0])
        pulse_width = float(
            np.asarray(handle["config/radar/pulse_width_s"]).reshape(-1)[0]
        )
        pri = float(np.asarray(handle["config/radar/pri_s"]).reshape(-1)[0])
        doppler_range = np.asarray(
            handle["config/params/VFT/doppler_Hz"]
        ).reshape(-1)
        result["physical_resolution_audit"] = {
            "sampling_rate_hz": fs,
            "pulse_width_s": pulse_width,
            "pri_s": pri,
            "active_pulse_samples": int(round(fs * pulse_width)),
            "pri_samples": int(round(fs * pri)),
            "active_pulse_fourier_resolution_hz": float(1.0 / pulse_width),
            "full_pri_fourier_resolution_hz": float(1.0 / pri),
            "vft_doppler_range_hz": [float(value) for value in doppler_range],
            "doppler_span_over_active_pulse_resolution": float(
                (doppler_range.max() - doppler_range.min()) / (1.0 / pulse_width)
            ),
        }
    result.update({"status": "completed", "locked_test_used": False})
    _write_json(Path(output_path), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "student_b"
        / "supervised_joint_six_class.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "student_b"
        / "oracle_parameter_specialists_20260723",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "student_b"
        / "oracle_parameter_specialists_20260723"
        / "stratified_diagnostics.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.config, args.output_root, args.output)
