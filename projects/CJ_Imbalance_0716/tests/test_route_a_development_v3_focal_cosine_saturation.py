import csv

from experiments.run_route_a_development_v3_focal_cosine_saturation import _plateau_from_history


def _write_history(path, values, lrs):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "validation_common_macro_aupr", "classifier_lr"
        ])
        writer.writeheader()
        for index, (value, lr) in enumerate(zip(values, lrs), start=1):
            writer.writerow({"epoch": index, "validation_common_macro_aupr": value,
                             "classifier_lr": lr})


def test_plateau_requires_small_tail_change_and_low_final_lr(tmp_path):
    path = tmp_path / "history.csv"
    values = [0.8] * 60 + [0.9 + index * 0.00002 for index in range(12)]
    lrs = [5e-4] * 71 + [5e-6]
    _write_history(path, values, lrs)
    result = _plateau_from_history(path)
    assert result["passed"]


def test_plateau_rejects_material_tail_gain(tmp_path):
    path = tmp_path / "history.csv"
    values = [0.8] * 60 + [0.9 + index * 0.0001 for index in range(12)]
    lrs = [5e-4] * 71 + [5e-6]
    _write_history(path, values, lrs)
    assert not _plateau_from_history(path)["passed"]
