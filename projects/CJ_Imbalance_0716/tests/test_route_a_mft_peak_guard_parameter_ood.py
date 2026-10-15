from experiments.audit_route_a_mft_peak_guard_ood_dataset import LEVELS, REGIMES
from experiments.run_route_a_u2_mft_peak_guard_parameter_ood_frozen import SUPPORTED_OOD


def test_ood_protocol_separates_supported_claim_from_low_count_boundary():
    assert REGIMES == ("ID_control", "low_count", "close_spacing", "ultra_close", "weak_target")
    assert SUPPORTED_OOD == ("close_spacing", "ultra_close", "weak_target")
    assert "low_count" not in SUPPORTED_OOD
    assert LEVELS == (25, 35, 50)

