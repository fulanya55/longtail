"""Audit sample, label, parameter-column, and component-metadata alignment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.parameter_schema import get_parameter_specs  # noqa: E402
from cjlt.data.registry import SIX_CLASS_NAMES  # noqa: E402
from experiments.run_shared import _data_path, _write_json, load_config  # noqa: E402


def _decode_matlab_char(handle, reference):
    values = np.asarray(handle[reference])
    return "".join(chr(int(value)) for value in values.reshape(-1, order="F"))


def _cell_numeric(handle, dataset, sample_index):
    reference = dataset[0, int(sample_index)]
    return np.asarray(handle[reference], dtype=np.float64).reshape(-1, order="F")


def audit_alignment(config_path, output_path, waveform_anchor_count=64):
    config = load_config(Path(config_path))
    data_path = _data_path(config)
    with h5py.File(str(data_path), "r") as handle:
        labels_selected = np.asarray(handle["label_selected"]).T.astype(np.uint8)
        labels_known = np.asarray(handle["known_label"]).T.astype(np.uint8)
        labels_all = np.asarray(handle["label_all"]).T.astype(np.uint8)
        parameter_values = np.asarray(handle["parameter_values"]).T.astype(np.float64)
        parameter_mask = np.asarray(handle["parameter_mask"]).T.astype(bool)
        registry_codes = [
            _decode_matlab_char(handle, reference)
            for reference in handle["registry_snapshot/code"][:, 0]
        ]
        selected_registry_indices = [registry_codes.index(name) for name in SIX_CLASS_NAMES]

        anchor_indices = np.linspace(
            0,
            len(labels_selected) - 1,
            num=min(int(waveform_anchor_count), len(labels_selected)),
            dtype=np.int64,
        )
        legacy = np.asarray(handle["data_legacy"][:, anchor_indices]).T.reshape(
            len(anchor_indices), 2, 3000
        )
        stored_complex = np.asarray(handle["iq_complex"][:, anchor_indices]).T
        complex_values = stored_complex["real"] + 1j * stored_complex["imag"]
        waveform_difference = max(
            float(np.max(np.abs(legacy[:, 0] - complex_values.real))),
            float(np.max(np.abs(legacy[:, 1] - complex_values.imag))),
        )
        anchor_hash = hashlib.sha256()
        anchor_hash.update(anchor_indices.tobytes())
        anchor_hash.update(labels_selected[anchor_indices].tobytes())
        anchor_hash.update(parameter_mask[anchor_indices].tobytes())
        anchor_hash.update(np.nan_to_num(parameter_values[anchor_indices]).tobytes())
        anchor_hash.update(legacy.astype(np.float64).tobytes())

        parameter_checks = {}
        all_parameter_checks_passed = True
        for class_index, class_name in enumerate(SIX_CLASS_NAMES):
            positive = labels_selected[:, class_index] == 1
            for spec in get_parameter_specs(class_name):
                mask = parameter_mask[:, spec.column_index]
                values = parameter_values[:, spec.column_index]
                key = "{}.{}".format(class_name, spec.name)
                entry = {
                    "column_index_zero_based": int(spec.column_index),
                    "positive_count": int(positive.sum()),
                    "mask_count": int(mask.sum()),
                    "label_mask_mismatch": int(np.logical_xor(positive, mask).sum()),
                    "valid_nonfinite": int(np.logical_and(mask, ~np.isfinite(values)).sum()),
                    "valid_min": float(values[mask].min()),
                    "valid_max": float(values[mask].max()),
                    "valid_unique": int(np.unique(values[mask]).size),
                }
                entry["passed"] = bool(
                    entry["label_mask_mismatch"] == 0
                    and entry["valid_nonfinite"] == 0
                )
                all_parameter_checks_passed &= entry["passed"]
                parameter_checks[key] = entry

        component_fields = (
            "component_seeds",
            "component_nominal_JNR_dB",
            "component_effective_JNR_dB",
            "component_nominal_amplitude",
            "component_effective_amplitude",
            "component_power_offset_dB",
            "component_measured_power_dB",
            "component_target_power_dB",
            "relative_power_dB",
            "weak_component_mask",
        )
        component_length_mismatch = {name: 0 for name in component_fields}
        jnr_mapping_max_abs_error = {"MFT": 0.0, "VFT": 0.0}
        jnr_mapping_count = {"MFT": 0, "VFT": 0}
        class_to_parameter_column = {"MFT": 27, "VFT": 48}
        for sample_index in range(len(labels_selected)):
            active = np.flatnonzero(labels_all[sample_index])
            cached = {}
            for field_name in component_fields:
                values = _cell_numeric(handle, handle[field_name], sample_index)
                cached[field_name] = values
                if len(values) != len(active):
                    component_length_mismatch[field_name] += 1
            nominal_jnr = cached["component_nominal_JNR_dB"]
            for class_name, parameter_column in class_to_parameter_column.items():
                registry_index = registry_codes.index(class_name)
                if labels_all[sample_index, registry_index] == 0:
                    continue
                active_position = int(np.flatnonzero(active == registry_index)[0])
                error = abs(
                    float(nominal_jnr[active_position])
                    - float(parameter_values[sample_index, parameter_column])
                )
                jnr_mapping_max_abs_error[class_name] = max(
                    jnr_mapping_max_abs_error[class_name], error
                )
                jnr_mapping_count[class_name] += 1

    selected_from_all = labels_all[:, selected_registry_indices]
    checks = {
        "known_label_equals_label_selected": bool(
            np.array_equal(labels_known, labels_selected)
        ),
        "selected_registry_projection_matches": bool(
            np.array_equal(selected_from_all, labels_selected)
        ),
        "legacy_iq_matches_complex_anchors": bool(waveform_difference == 0.0),
        "all_parameter_label_masks_match": bool(all_parameter_checks_passed),
        "all_component_lengths_match_active_labels": bool(
            all(value == 0 for value in component_length_mismatch.values())
        ),
        "mft_vft_component_jnr_matches_parameter_columns": bool(
            all(value == 0.0 for value in jnr_mapping_max_abs_error.values())
        ),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "data_path": str(data_path),
        "sample_count": int(len(labels_selected)),
        "selected_class_order": list(SIX_CLASS_NAMES),
        "registry_codes": registry_codes,
        "selected_registry_indices_zero_based": selected_registry_indices,
        "checks": checks,
        "waveform_anchor_count": int(len(anchor_indices)),
        "waveform_anchor_indices_sha256": hashlib.sha256(
            anchor_indices.tobytes()
        ).hexdigest(),
        "waveform_anchor_content_sha256": anchor_hash.hexdigest(),
        "legacy_iq_max_abs_difference": waveform_difference,
        "parameter_checks": parameter_checks,
        "component_length_mismatch_samples": component_length_mismatch,
        "component_jnr_mapping_count": jnr_mapping_count,
        "component_jnr_mapping_max_abs_error": jnr_mapping_max_abs_error,
        "locked_test_used": False,
    }
    _write_json(Path(output_path), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


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
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "student_b"
        / "oracle_parameter_specialists_20260723"
        / "alignment_audit.json",
    )
    parser.add_argument("--waveform-anchor-count", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit_alignment(args.config, args.output, args.waveform_anchor_count)
