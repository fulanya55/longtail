from __future__ import annotations

from experiments.audit_route_a_pgl_compatibility import COMMIT, PATH, REPOSITORY


def test_pgl_audit_is_pinned_to_one_official_source_revision() -> None:
    assert REPOSITORY == "Kunmonkey/PGLoss"
    assert len(COMMIT) == 40
    assert PATH.endswith("pg_loss.py")
