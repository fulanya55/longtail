from __future__ import annotations

import json

from experiments.audit_route_a_frequency_group_visibility_invariance import run


def test_frequency_losses_are_directly_visibility_blind(tmp_path) -> None:
    manifest = tmp_path / "seal.json"
    manifest.write_text(json.dumps({"test_v3": {"access_authorized": False, "analysis_allowed": False, "used": False}}), encoding="utf-8")
    payload = run(tmp_path / "summary.json", manifest)
    assert payload["checks"]["every_loss_is_mask_swap_invariant"]
    assert payload["checks"]["every_loss_gradient_is_mask_swap_invariant"]
    assert "DB" in payload["same_label_equal_logit_gradient_summary"]["cooccurrence_sensitive_losses"]
    assert payload["training_performed"] is False
    assert payload["test_v3_used"] is False
