import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/audit_route_a_frequency_v2_stage1.py"
SPEC = importlib.util.spec_from_file_location("frequency_v2_stage1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def balanced_labels() -> np.ndarray:
    rows = []
    for value in range(1, 64):
        row = np.asarray([(value >> bit) & 1 for bit in range(6)], dtype=np.uint8)
        rows.extend([row] * 400)
    return np.asarray(rows, dtype=np.uint8)


def test_preregistered_role_contract_covers_all_identities():
    roles = MODULE.build_roles(balanced_labels())
    audit = MODULE.audit_roles(roles, 25200)
    assert audit["counts"] == {
        "optimization": 15120,
        "early": 2520,
        "calibration": 2520,
        "threshold": 2520,
        "evaluation": 2520,
    }
    assert audit["covers_all_identities"]
    assert audit["all_pairwise_disjoint"]


def test_preregistered_manifest_targets_are_frozen():
    assert MODULE.FB_TARGET.tolist() == [1200] * 6
    assert MODULE.LABEL_SPECS["if20"][1].tolist() == [1500, 75, 824, 453, 249, 137]
    assert MODULE.COMBO_SPECS["if50"] == (3930100023, 50.0)


def test_manifest_if_uses_positive_count_extremes():
    label_payload = {"achieved_label_counts": [100, 10, 80, 60, 40, 20]}
    combo_payload = {"combo_counts": {"a": 240, "b": 24, "c": 100}}
    assert MODULE.manifest_if(label_payload, "label") == 10.0
    assert MODULE.manifest_if(combo_payload, "combo") == 10.0


def test_receiver_seed_mapping_is_bijective_for_preregistered_range():
    sample = np.asarray(
        [MODULE.EXPECTED_SAMPLE_SEED_FIRST, MODULE.EXPECTED_SAMPLE_SEED_LAST],
        dtype=np.int64,
    )
    receiver = (sample + MODULE.RECEIVER_SEED_SALT) % MODULE.UINT32_MODULUS
    recovered = (receiver - MODULE.RECEIVER_SEED_SALT) % MODULE.UINT32_MODULUS
    assert receiver.tolist() == [4128677743, 4128702942]
    assert np.array_equal(recovered, sample)
