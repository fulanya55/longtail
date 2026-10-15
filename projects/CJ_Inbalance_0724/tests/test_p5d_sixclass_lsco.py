import torch

from cj0724.audit_p5d_lsco import audit_rows, promotion_gates
from cj0724.p5d_sixclass_lsco import class_audit


def test_class_audit_reports_fixed_threshold_recall_and_fpr():
    outputs = {
        "logits": torch.tensor([[8.0, -8.0, -8.0, -8.0, -8.0, -8.0],
                                [-8.0, 8.0, -8.0, -8.0, -8.0, -8.0],
                                [8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]),
        "labels": torch.tensor([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0]]),
    }

    audit = class_audit(outputs)

    assert audit["FMB"]["recall"] == 1.0
    assert audit["FMB"]["fpr"] == 0.5
    assert audit["SawFM"]["recall"] == 0.5
    assert audit["SawFM"]["fpr"] == 0.0


def test_p5d_audit_requires_complete_safe_pairs(tmp_path):
    rows = []
    per_class = {name: {"recall": 0.5, "fpr": 0.01} for name in ("FMB", "SawFM")}
    base_metrics = {
        "parameter": {"oracle_macro_target_nmae": 1.0, "per_target": [{"class_name": "MFT", "name": "jnr", "normalized_mae": 0.6}]},
        "macro_aupr": 0.5,
        "subset_acc": 0.5,
    }
    candidate_metrics = {
        "parameter": {"oracle_macro_target_nmae": 0.9, "per_target": [{"class_name": "MFT", "name": "jnr", "normalized_mae": 0.5}]},
        "macro_aupr": 0.501,
        "subset_acc": 0.5,
    }
    for fold in range(3):
        for seed in (15041, 15042, 15043):
            for model, metrics, recall in (
                ("legacy_mlcnn_global", base_metrics, 0.5),
                ("mlcnn_iq_grouped_task_routed", candidate_metrics, 0.6),
            ):
                checkpoint = tmp_path / f"{fold}_{seed}_{model}.pt"
                checkpoint.touch()
                rows.append({
                    "fold": fold,
                    "seed": seed,
                    "model": model,
                    "checkpoint": str(checkpoint),
                    "unseen": {
                        "metrics": metrics,
                        "per_class": {name: {"recall": recall, "fpr": 0.01} for name in per_class},
                        "logits_sha256": "a" * 64,
                    },
                })

    _, missing_rows, missing_fields, paired = audit_rows(rows)
    gates = promotion_gates(paired, not missing_rows and not missing_fields)

    assert not missing_rows
    assert not missing_fields
    assert gates["all_pass"]

    paired[0]["class_deltas"]["FMB"] = {"recall": 0.01, "fpr": 0.021}
    assert not promotion_gates(paired, True)["gates"]["no_recall_gain_with_fpr_increase_over_0.02"]
