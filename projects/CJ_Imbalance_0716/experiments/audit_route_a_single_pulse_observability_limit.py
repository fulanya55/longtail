"""Audit the physical observability limits of the frozen uniform-PRI 0716 protocol.

This is deliberately a configuration-only audit.  It never opens I/Q, labels,
predictions, checkpoints, calibration files, or any test set.  It quantifies
two necessary (not sufficient) constraints that explain why an additional
single-PRI MFT/VFT expert cannot be claimed merely by increasing capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(
    r"C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成\create_closedset_6class_config.m"
)
DEFAULT_SEAL = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/tapc/route_a_single_pulse_observability_limit/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _range(text: str, field: str) -> tuple[float, float]:
    match = re.search(
        rf"cfg\.params\.{re.escape(field)}\s*=\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]",
        text,
    )
    if match is None:
        raise ValueError(f"cannot parse range for {field}")
    return float(match.group(1)), float(match.group(2))


def _scalar(text: str, field: str) -> float:
    match = re.search(rf"cfg\.radar\.{re.escape(field)}\s*=\s*([0-9.eE+-]+)", text)
    if match is None:
        raise ValueError(f"cannot parse scalar for radar.{field}")
    return float(match.group(1))


def calculate(text: str) -> dict:
    fs = _scalar(text, "fs_Hz")
    pulse_width = _scalar(text, "pulse_width_s")
    pri = _scalar(text, "pri_s")
    mft_delay = _range(text, "MFT.target_delay_s")
    mft_targets = _range(text, "MFT.num_targets")
    vft_doppler = _range(text, "VFT.doppler_Hz")
    delay_span = mft_delay[1] - mft_delay[0]
    active_resolution = 1.0 / pulse_width
    max_vft_cycles = max(abs(value) for value in vft_doppler) * pulse_width
    min_vft_cycles = min(abs(value) for value in vft_doppler) * pulse_width
    return {
        "radar": {
            "sampling_rate_hz": fs,
            "pulse_width_s": pulse_width,
            "pri_s": pri,
            "active_pulse_samples": int(round(fs * pulse_width)),
            "pri_samples": int(round(fs * pri)),
            "active_pulse_fourier_resolution_hz": active_resolution,
        },
        "mft": {
            "target_count_range": [int(mft_targets[0]), int(mft_targets[1])],
            "relative_delay_range_s": list(mft_delay),
            "largest_possible_pairwise_delay_s": delay_span,
            "pulse_support_overlap_lower_bound_fraction": 1.0 - delay_span / pulse_width,
            "interpretation": (
                "Every pair of MFT copies has at least this fractional pulse-support overlap; "
                "the audit does not infer per-sample delays or prove classification impossibility."
            ),
        },
        "vft": {
            "doppler_range_hz": list(vft_doppler),
            "doppler_span_hz": vft_doppler[1] - vft_doppler[0],
            "span_over_active_pulse_fourier_resolution": (vft_doppler[1] - vft_doppler[0]) / active_resolution,
            "minimum_phase_cycles_over_active_pulse": min_vft_cycles,
            "maximum_phase_cycles_over_active_pulse": max_vft_cycles,
            "interpretation": (
                "This is the single active-LFM-pulse aperture only.  The audit does not use the "
                "noise-only remainder of the PRI as extra coherent VFT observation time."
            ),
        },
    }


def run(config: Path, seal: Path, output: Path) -> dict:
    if not config.is_file():
        raise FileNotFoundError(config)
    if not seal.is_file():
        raise FileNotFoundError(seal)
    sealed = json.loads(seal.read_text(encoding="utf-8-sig"))
    test_v3 = sealed["test_v3"]
    if any(test_v3[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    values = calculate(config.read_text(encoding="utf-8-sig"))
    checks = {
        "mft_support_overlap_lower_bound_ge_0p80": values["mft"]["pulse_support_overlap_lower_bound_fraction"] >= 0.80,
        "vft_doppler_span_less_than_one_active_pulse_bin": values["vft"]["span_over_active_pulse_fourier_resolution"] < 1.0,
        "vft_maximum_phase_cycles_le_one": values["vft"]["maximum_phase_cycles_over_active_pulse"] <= 1.0,
        "test_v3_seal_intact": True,
    }
    result = {
        "status": "completed_configuration_only_observability_audit",
        "experiment": "RouteA_SinglePulse_MFT_VFT_ObservabilityLimit",
        "evidence_role": "protocol_constraint_audit_not_model_evidence",
        "source_config": str(config),
        "source_config_sha256": sha256(config),
        "test_v3_manifest": str(seal),
        "values": values,
        "checks": checks,
        "decision": {
            "single_pulse_mft_vft_capacity_retuning_permitted": False,
            "required_before_any_new_mft_experiment": "new delay-aperture and minimum-separation protocol",
            "required_before_any_new_vft_experiment": "new multi-PRI coherent-aperture protocol",
            "does_not_prove": [
                "Bayes impossibility on the original 0716 task",
                "a model or loss improvement",
                "a test-v3 result",
            ],
        },
        "iq_opened": False,
        "labels_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "threshold_selection_performed": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--test-v3-manifest", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.config, args.test_v3_manifest, args.output), ensure_ascii=False, indent=2))
