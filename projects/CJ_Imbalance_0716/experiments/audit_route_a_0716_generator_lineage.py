"""Statically audit that the actual 0716 generator matches the stored protocol.

The audit intentionally reads MATLAB source and the test-v3 seal only.  It does
not open generated signals, labels, parameter values, predictions, or any test
data.  Historical reference generators elsewhere under C:\\LJQ_code are out of
scope because the official 0716 generator path is explicitly pinned here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = Path(r"C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成")
DEFAULT_CONFIG = GENERATOR_ROOT / "create_closedset_6class_config.m"
DEFAULT_MFT = GENERATOR_ROOT / "generators/generate_mft.m"
DEFAULT_VFT = GENERATOR_ROOT / "generators/generate_vft.m"
DEFAULT_SEAL = ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/tapc/route_a_0716_generator_lineage/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contains_all(text: str, *phrases: str) -> bool:
    return all(phrase in text for phrase in phrases)


def audit(config: Path, mft: Path, vft: Path, seal: Path, output: Path) -> dict:
    for path in (config, mft, vft, seal):
        if not path.is_file():
            raise FileNotFoundError(path)
    seal_value = json.loads(seal.read_text(encoding="utf-8-sig"))["test_v3"]
    if any(seal_value[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    config_text = config.read_text(encoding="utf-8-sig")
    mft_text = mft.read_text(encoding="utf-8-sig")
    vft_text = vft.read_text(encoding="utf-8-sig")
    checks = {
        "config_declares_single_pri_100us_lfm": contains_all(
            config_text,
            "cfg.radar.pulse_width_s = 100e-6",
            "cfg.radar.pri_s = 300e-6",
            "cfg.radar.num_pulses = 1",
        ),
        "config_declares_mft_3_to_8_and_30_to_50us": contains_all(
            config_text,
            "cfg.params.MFT.num_targets = [3, 8]",
            "cfg.params.MFT.target_delay_s = [30e-6, 50e-6]",
        ),
        "config_declares_vft_1_to_7khz": "cfg.params.VFT.doppler_Hz = [1e3, 7e3]" in config_text,
        "mft_samples_unique_integer_delay_slots": contains_all(
            mft_text, "sample_unique_delay_samples", "randperm(stream, slotCount, count) - 1"
        ),
        "mft_sorts_and_records_final_delays": contains_all(
            mft_text, "[relativeDelaySamples, order] = sort", "delaySamples = relativeDelaySamples + arrivalDelaySamples"
        ),
        "mft_validates_complete_pri_support": contains_all(
            mft_text, "validate_delay_identifiability", "最大总时延加脉宽必须完整位于同一 PRI 内"
        ),
        "vft_uses_lfm_phase_modulation_and_arrival_delay": contains_all(
            vft_text, "reference.train .* exp", "doppler_Hz * reference.time_train_s", "apply_arrival_delay"
        ),
        "test_v3_seal_intact": True,
    }
    result = {
        "status": "completed_static_generator_lineage_audit",
        "experiment": "RouteA_0716_GeneratorLineage",
        "evidence_role": "source_semantics_and_protocol_provenance_not_model_evidence",
        "official_generator_root": str(GENERATOR_ROOT),
        "sources": {
            "closedset_config": {"path": str(config), "sha256": sha256(config)},
            "mft_generator": {"path": str(mft), "sha256": sha256(mft)},
            "vft_generator": {"path": str(vft), "sha256": sha256(vft)},
        },
        "checks": checks,
        "decision": {
            "historical_reference_generator_reused_as_0716_evidence": False,
            "mft_semantics": "unique sampled and sorted final delays with configured finite support",
            "vft_semantics": "single-PRI LFM multiplied by Doppler phase then shifted by arrival delay",
            "does_not_prove": ["dataset bitwise reconstruction", "model accuracy", "physical observability beyond the configured aperture"],
        },
        "iq_opened": False,
        "labels_opened": False,
        "data_generated": False,
        "training_performed": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    if not all(checks.values()):
        raise ValueError(f"generator lineage check failed: {checks}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mft-generator", type=Path, default=DEFAULT_MFT)
    parser.add_argument("--vft-generator", type=Path, default=DEFAULT_VFT)
    parser.add_argument("--test-v3-manifest", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(audit(args.config, args.mft_generator, args.vft_generator, args.test_v3_manifest, args.output), ensure_ascii=False, indent=2))
