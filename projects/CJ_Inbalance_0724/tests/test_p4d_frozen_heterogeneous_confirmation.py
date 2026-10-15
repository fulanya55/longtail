from cj0724.p4d_frozen_heterogeneous_confirmation import audit_promotion, paired_comparisons


def _evaluation(data_seed, model_seed, model, nmae, pearson, aupr, subset, saw_delta=0.0):
    base_targets = [
        {"class_name": "SawFM", "name": "sweep_period", "normalized_mae": 0.8},
        {"class_name": "ISRJ", "name": "sample_width", "normalized_mae": 0.7},
    ]
    targets = [dict(row) for row in base_targets]
    if model == "mlcnn_iq_grouped_task_routed":
        targets[0]["normalized_mae"] += saw_delta
        targets[1]["normalized_mae"] -= 0.1
    return {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "model": model,
        "metrics": {
            "parameter_macro_nmae": nmae,
            "parameter_macro_pearson": pearson,
            "macro_aupr": aupr,
            "subset_acc": subset,
        },
        "per_target": targets,
    }


def test_pairing_and_promotion_gate_pass():
    evaluations = []
    for data_seed in (1, 2):
        for model_seed in (11, 12, 13):
            evaluations.append(_evaluation(data_seed, model_seed, "legacy_mlcnn_global", 1.0, 0.4, 0.9, 0.5))
            evaluations.append(_evaluation(data_seed, model_seed, "mlcnn_iq_grouped_task_routed", 0.94, 0.5, 0.899, 0.49, 0.01))
    pairs = paired_comparisons(evaluations)
    preregister = {"promotion_gates": {
        "each_of_6_paired_route_parameter_macro_nmae_reduction_at_least": 0.04,
        "mean_route_parameter_macro_nmae_reduction_at_least": 0.05,
        "each_paired_route_macro_aupr_drop_at_most": 0.003,
        "each_paired_route_subset_accuracy_drop_at_most": 0.03,
        "each_paired_route_parameter_macro_pearson_not_lower": True,
        "no_target_worse_in_all_3_model_seeds_within_either_data_seed_by_more_than_absolute_nmae": 0.03,
    }}
    audit = audit_promotion(pairs, preregister)
    assert len(pairs) == 6
    assert audit["promotion_passed"] is True


def test_systematic_target_failure_is_detected():
    evaluations = []
    for model_seed in (11, 12, 13):
        evaluations.append(_evaluation(1, model_seed, "legacy_mlcnn_global", 1.0, 0.4, 0.9, 0.5))
        evaluations.append(_evaluation(1, model_seed, "mlcnn_iq_grouped_task_routed", 0.9, 0.5, 0.9, 0.5, 0.04))
    pairs = paired_comparisons(evaluations)
    preregister = {"promotion_gates": {
        "each_of_6_paired_route_parameter_macro_nmae_reduction_at_least": 0.04,
        "mean_route_parameter_macro_nmae_reduction_at_least": 0.05,
        "each_paired_route_macro_aupr_drop_at_most": 0.003,
        "each_paired_route_subset_accuracy_drop_at_most": 0.03,
        "each_paired_route_parameter_macro_pearson_not_lower": True,
        "no_target_worse_in_all_3_model_seeds_within_either_data_seed_by_more_than_absolute_nmae": 0.03,
    }}
    audit = audit_promotion(pairs, preregister)
    assert audit["promotion_passed"] is False
    assert audit["systematic_target_failures"][0]["target"] == "SawFM.sweep_period"
