from cj0724.p2a_preflight import validate_reference


def valid_payloads():
    method = {
        "test_used_at_freeze": False,
        "seed_freezes": [{"seed": 2041}, {"seed": 2042}, {"seed": 2043}],
    }
    summary = {
        "method_freeze_sha256": "abc",
        "level3_validated": True,
        "locked_test_used": False,
        "test_used_once": True,
        "test_fit_or_selection": False,
    }
    return method, summary


def test_validate_reference_accepts_frozen_p1n_boundary():
    method, summary = valid_payloads()
    assert validate_reference(method, summary, "abc", "abc") == []


def test_validate_reference_rejects_hash_and_test_leakage():
    method, summary = valid_payloads()
    summary["test_fit_or_selection"] = True
    errors = validate_reference(method, summary, "actual", "registered")
    assert any("hash differs" in error for error in errors)
    assert any("fitting or selection" in error for error in errors)
