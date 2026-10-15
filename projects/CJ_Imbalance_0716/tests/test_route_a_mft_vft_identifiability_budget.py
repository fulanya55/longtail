from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.audit_route_a_mft_vft_identifiability_budget import audit


def _train_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        radar = handle.create_group("config/radar")
        radar["fs_Hz"] = [10e6]; radar["pulse_width_s"] = [100e-6]
        radar["pri_s"] = [300e-6]; radar["num_pulses"] = [1]
        mft = handle.create_group("config/params/MFT")
        mft["num_targets"] = [3, 8]; mft["target_delay_s"] = [30e-6, 50e-6]
        vft = handle.create_group("config/params/VFT")
        vft["doppler_Hz"] = [1e3, 7e3]


def test_reproduces_budget_from_train_metadata(tmp_path: Path) -> None:
    source, output = tmp_path / "train.mat", tmp_path / "budget.json"
    _train_h5(source)
    value = audit(source, output)
    assert output.exists()
    assert value["contract_checks"] == {
        "single_pulse": True,
        "vft_span_below_single_pulse_fourier_scale": True,
        "mft_delay_span_below_pulse_width": True,
        "mft_overlap_at_least_0p80": True,
    }
    assert np.isclose(value["vft"]["single_pulse_fourier_scale_hz"], 10000.0)
    assert np.isclose(value["mft"]["overlap_lower_bound_fraction"], 0.8)


def test_rejects_test_named_input(tmp_path: Path) -> None:
    source = tmp_path / "test.mat"; _train_h5(source)
    with pytest.raises(ValueError, match="train-only"):
        audit(source, tmp_path / "budget.json")
