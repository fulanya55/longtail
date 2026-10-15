import json
from pathlib import Path

from experiments.audit_route_a_mft_peak_count_development_v9_frozen import (
    CANDIDATE_SCORE,
    EXPECTED_SOURCE_SHA,
)
from experiments.run_route_a_development_v2_h5 import _sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_candidate_and_source_hash_are_unchanged():
    manifest = json.loads(
        (PROJECT_ROOT / "frozen/20260801_route_a_mft_peak_count_v1/mft_peak_count_frozen_manifest.json").read_text(encoding="utf-8")
    )
    source = PROJECT_ROOT / "experiments/audit_route_a_mft_pulse_compression_morphology_v3.py"
    assert CANDIDATE_SCORE == "nms_count_gt_bg_q99"
    assert manifest["feature_contract"]["learned_parameters"] == 0
    assert _sha256(source).lower() == EXPECTED_SOURCE_SHA

