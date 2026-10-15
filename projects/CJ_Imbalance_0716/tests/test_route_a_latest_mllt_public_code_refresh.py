from __future__ import annotations

from experiments import audit_route_a_latest_mllt_public_code_refresh as audit


def test_cross_query_refresh_preserves_non_approximation_boundary(tmp_path, monkeypatch) -> None:
    def fake_get_json(url: str, github: bool = False) -> dict:
        if "crossref" in url:
            return {"message": {"title": [audit.TITLE], "author": [{"family": "Chen"}]}}
        return {"total_count": 0, "items": []}

    monkeypatch.setattr(audit, "_get_json", fake_get_json)
    payload = audit.run(tmp_path / "receipt.json")

    assert set(payload["github_cross_query_searches"]) == set(audit.GITHUB_QUERY_TERMS)
    assert payload["github_exact_title_repository_count"] == 0
    assert payload["official_or_auditable_code_located"] is False
    assert "do_not_implement" in payload["decision"]
    assert payload["locked_test_used"] is False
    assert payload["test_v3_used"] is False
