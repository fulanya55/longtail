from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.seal_route_a_test_v3 import identity_audit, read_receiver_ids


def _write_ids(path: Path, values):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("receiver_noise_seed", data=np.asarray(values)[None, :])


def test_identity_audit_accepts_unique_disjoint_roles(tmp_path):
    development = tmp_path / "development.mat"
    test = tmp_path / "test.mat"
    _write_ids(development, [1, 2, 3])
    _write_ids(test, [11, 12, 13])

    result = identity_audit(development, test)

    assert result["passed"] is True
    assert result["cross_role_receiver_id_intersection_count"] == 0


@pytest.mark.parametrize(
    "development_ids,test_ids",
    [([1, 1, 2], [11, 12, 13]), ([1, 2, 3], [3, 4, 5])],
)
def test_identity_audit_rejects_duplicate_or_cross_role_identity(
    tmp_path, development_ids, test_ids
):
    development = tmp_path / "development.mat"
    test = tmp_path / "test.mat"
    _write_ids(development, development_ids)
    _write_ids(test, test_ids)

    assert identity_audit(development, test)["passed"] is False


def test_read_receiver_ids_rejects_missing_field(tmp_path):
    path = tmp_path / "missing.mat"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(KeyError):
        read_receiver_ids(path)
