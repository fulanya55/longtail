from experiments.run_route_a_h1_b2_mft_vft_development_v12_frozen import SHARED_VARIANT, _promote_shared_rows
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import VARIANTS


def test_shared_variant_name_is_distinct():
    assert SHARED_VARIANT not in VARIANTS


def test_promote_shared_rows_replaces_old_q4():
    rows = [{"variants": [{"variant": VARIANTS[4], "value": 1}, {"variant": SHARED_VARIANT, "value": 2}]}]
    promoted = _promote_shared_rows(rows)
    q4 = [item for item in promoted[0]["variants"] if item["variant"] == VARIANTS[4]]
    assert len(q4) == 1
    assert q4[0]["value"] == 2
    assert rows[0]["variants"][1]["variant"] == SHARED_VARIANT
