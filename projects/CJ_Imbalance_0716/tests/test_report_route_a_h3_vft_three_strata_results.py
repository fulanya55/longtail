from argparse import Namespace
import json

from experiments.report_route_a_h3_vft_three_strata_results import run


def test_h3_report_preserves_development_only_stop_boundary(tmp_path):
    common = {"external_used": False, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    observable = {**common, "status": "completed_no_fit_observable", "trained_parameters": 0,
                  "weak_vs_absent": {"true_auroc": .7, "permuted_auroc": .5, "true_minus_permuted_auroc": .2},
                  "strong_vs_absent": {"true_auroc": .8, "permuted_auroc": .5},
                  "bootstrap": {"weak_true_minus_permuted_auroc_95_ci": [.1, .3]},
                  "gate": {"passed": True}}
    delta = {"mean": .01, "bootstrap_95_ci": [-.01, .03], "positive_seeds": 3}
    summary = {**common, "status": "completed_development_summary", "gate": {"passed": False, "checks": {"fixed_gate": False}},
               "comparison": {"T1_minus_T2_weak_vft_auroc": delta, "T1_minus_T2_weak_vft_aupr": delta,
                              "T1_minus_T0_strong_vft_auroc": delta}}
    audit = {**common, "status": "completed_artifact_audit", "record_count": 15, "t1_t2_equal_parameter_count": True}
    paths = []
    for name, value in (("observable.json", observable), ("summary.json", summary), ("audit.json", audit)):
        path = tmp_path / name; path.write_text(json.dumps(value), encoding="utf-8"); paths.append(path)
    output = tmp_path / "report.md"
    run(Namespace(observable=paths[0], summary=paths[1], audit=paths[2], output=output))
    text = output.read_text(encoding="utf-8")
    assert "未通过 H3" in text
    assert "不改变时间戳排列" in text
    assert "test_v3_used=false" in text
