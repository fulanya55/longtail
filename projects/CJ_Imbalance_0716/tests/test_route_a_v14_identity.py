import pytest

from experiments.route_a_v14_identity import RECEIVER_RANGE, disjoint, preflight


def test_v14_preflight_is_disjoint_and_never_opens_tests():
    result = preflight()
    assert result["identity_audit_passed"] is True
    assert all(result["receiver_identity_disjoint"].values())
    assert result["test_files_opened"] is False


def test_disjoint_rejects_an_overlapping_identity():
    assert disjoint(RECEIVER_RANGE, (342677743, 342702942)) is False
    assert disjoint(RECEIVER_RANGE, (352677743, 352702942)) is True


def test_train_audit_refuses_test_named_path():
    from experiments.route_a_v14_identity import audit_train
    with pytest.raises(ValueError, match="test-named"):
        audit_train("C:/example/closedset_6class_test.mat")
