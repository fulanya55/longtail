"""Freeze development configuration without opening locked test files."""

import argparse
import hashlib
import json
from pathlib import Path


def build_freeze_payloads():
    common = {"allow_locked_test": False}
    return {
        "frozen_data_config.json": dict(
            common,
            data_root=r"C:\LJQ_code\干扰数据\复合干扰数据0716",
            train_file="closedset_6class_train.mat",
            locked_test_file="closedset_6class_test.mat",
            power_sweep_dir="power_sweep_9level_v1",
            class_modes={"pilot": "four", "formal": "six"},
            normalization="complex_rms",
        ),
        "frozen_split_config.json": dict(
            common,
            seed_list=[2026, 2027, 2028, 2029, 2030],
            roles=["optimization", "early_stop", "calibration", "threshold_selection"],
            labeled_ratios=[0.05, 0.10, 0.20, 0.50, 1.00],
            identity_key="receiver_noise_seed",
        ),
        "frozen_model_config.json": dict(
            common,
            backbone="complex_1d_true",
            default_width=16,
            input_shape=[2, 3000],
            classifier_path="original_logits_head",
        ),
        "frozen_training_config.json": dict(
            common,
            optimizer="AdamW",
            learning_rate=3e-4,
            weight_decay=1e-4,
            classification_loss="CB-ASL",
            checkpoint_order=["subset_accuracy", "hamming_loss", "earlier_epoch"],
        ),
        "frozen_calibration_config.json": dict(
            common,
            temperature="shared_scalar",
            source_role="calibration",
        ),
        "frozen_threshold_config.json": dict(
            common,
            methods=["global", "shrinkage_per_label"],
            source_role="threshold_selection",
            pseudo_tau_pos=0.95,
            pseudo_tau_neg=0.05,
        ),
        "frozen_student_a_config.json": dict(
            common,
            method="LCPT-CJNet",
            variants=["A0", "A2", "A3", "A4"],
            prototype_weight=0.2,
            power_weight=0.2,
            power_levels=[10, 15, 20, 25, 30, 35, 40, 45, 50],
        ),
        "frozen_student_b_config.json": dict(
            common,
            method="DAPS-CJNet-v2",
            variants=["B0", "B1", "B2", "B3", "B4", "B5", "B6"],
            teacher_momentum=[0.99, 0.999],
            supervised_parameter_weight=0.10,
            unsupervised_parameter_weight=0.20,
        ),
    }


def _write_json(path, payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_configuration_freeze(output_dir, payloads):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, payload in payloads.items():
        sha256 = _write_json(directory / name, payload)
        files[name] = {"sha256": sha256}
    manifest = {
        "status": "development_configuration_frozen",
        "locked_test_authorized": False,
        "files": files,
    }
    _write_json(directory / "configuration_freeze.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "frozen" / "20260721_pilot_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = write_configuration_freeze(args.output_dir, build_freeze_payloads())
    print(json.dumps(result, ensure_ascii=False, indent=2))
