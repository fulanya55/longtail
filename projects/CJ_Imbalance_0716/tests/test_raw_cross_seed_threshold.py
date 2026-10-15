from experiments.run_route_a_raw_cross_seed_threshold import _loso_selections


def _rows(offset=0.0):
    result = []
    for threshold in (0.4, 0.5, 0.6):
        fpr = {0.4: 0.18, 0.5: 0.12, 0.6: 0.08}[threshold] + offset
        result.append({
            "threshold": threshold,
            "ordinary_fpr": fpr,
            "max_power_fpr": fpr,
            "ordinary_recall": 1.0 - threshold / 2,
            "mean_power_recall": 1.0 - threshold / 10,
            "power": {"50": {"recall": 1.0 - threshold / 10}},
        })
    return result


def test_loso_excludes_held_out_seed():
    searches = {2039: _rows(0.0), 2040: _rows(0.01), 2041: _rows(0.02)}
    selected = _loso_selections(searches, 0.15)
    for seed, payload in selected.items():
        assert seed not in payload["source_seeds"]
        assert len(payload["source_seeds"]) == 2


def test_loso_selects_best_feasible_common_threshold():
    searches = {2039: _rows(0.0), 2040: _rows(0.01), 2041: _rows(0.02)}
    selected = _loso_selections(searches, 0.15)
    assert all(payload["threshold"] == 0.5 for payload in selected.values())

