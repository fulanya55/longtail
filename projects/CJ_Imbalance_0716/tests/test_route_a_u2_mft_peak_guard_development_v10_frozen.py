import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v10_guard_contract_is_one_sided_and_sealed():
    manifest = json.loads((PROJECT_ROOT / "frozen/20260801_route_a_u2_mft_peak_guard_v1/u2_mft_peak_guard_frozen_manifest.json").read_text(encoding="utf-8"))
    assert manifest["integration_contract"]["residual_cap"] == 0.75
    assert manifest["integration_contract"]["affected_logit"] == "MFT only"
    assert manifest["integration_contract"]["u2_thresholds_reused"] is True
    assert manifest["test_v3"] == {"access_authorized": False, "analysis_allowed": False, "used": False}

