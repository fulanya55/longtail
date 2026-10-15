from __future__ import annotations

import base64

from experiments import audit_route_a_public_mllt_candidate_triage as audit


def test_untraceable_public_repository_is_not_admitted(tmp_path, monkeypatch) -> None:
    encoded = base64.b64encode(b"title: an unreviewed public release\n").decode("ascii")

    def fake_get_json(url: str) -> dict:
        if "/git/trees/" in url:
            return {"tree": [{"path": path} for path in audit.CANDIDATES["tec_graph"]["interface_markers"]]}
        if "/contents/CITATION.cff" in url or "/contents/README.md" in url:
            return {"sha": "fakeblob", "content": encoded}
        return {"default_branch": "main", "html_url": "https://example.invalid/repo", "created_at": "2026-01-01", "updated_at": "2026-01-02", "license": {"spdx_id": None}}

    monkeypatch.setattr(audit, "_get_json", fake_get_json)
    original = audit.CANDIDATES
    monkeypatch.setattr(audit, "CANDIDATES", {"tec_graph": original["tec_graph"]})
    payload = audit.run(tmp_path / "receipt.json")

    row = payload["candidates"][0]
    assert row["citation_has_doi"] is False
    assert row["exact_interface_candidate"] is True
    assert row["exact_equal_protocol_reproduction_admissible"] is False
    assert payload["any_exact_equal_protocol_reproduction_admissible"] is False
    assert payload["test_v3_used"] is False
