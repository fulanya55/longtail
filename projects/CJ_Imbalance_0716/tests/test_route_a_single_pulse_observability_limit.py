import pytest

from experiments.audit_route_a_single_pulse_observability_limit import calculate


def test_uniform_pri_contract_has_expected_observability_constraints():
    text = """
    cfg.radar.fs_Hz = 10e6;
    cfg.radar.pulse_width_s = 100e-6;
    cfg.radar.pri_s = 300e-6;
    cfg.params.MFT.target_delay_s = [30e-6, 50e-6];
    cfg.params.MFT.num_targets = [3, 8];
    cfg.params.VFT.doppler_Hz = [1e3, 7e3];
    """
    result = calculate(text)
    assert result["mft"]["pulse_support_overlap_lower_bound_fraction"] == pytest.approx(0.8)
    assert result["vft"]["span_over_active_pulse_fourier_resolution"] == pytest.approx(0.6)
    assert result["vft"]["maximum_phase_cycles_over_active_pulse"] == pytest.approx(0.7)
