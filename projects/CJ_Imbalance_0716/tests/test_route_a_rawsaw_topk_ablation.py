from experiments.run_route_a_rawsaw_topk_ablation import NON_SAW_INDICES, TOPKS


def test_topk_ablation_covers_goal_values_and_keeps_sawfm_exclusive():
    assert TOPKS == (4, 8, 16)
    assert NON_SAW_INDICES == (0, 2, 3, 4, 5)
