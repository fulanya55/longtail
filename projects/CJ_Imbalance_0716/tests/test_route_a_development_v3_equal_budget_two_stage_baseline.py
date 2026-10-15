from argparse import Namespace

from experiments.run_route_a_development_v3_equal_budget_two_stage_baseline import _two_stage_args


def test_two_stage_budget_is_fixed_and_disables_early_stop():
    source = Namespace(epochs=36, patience=4, untouched="ok")
    derived = _two_stage_args(source, 24)
    assert source.epochs == 36
    assert source.patience == 4
    assert derived.epochs == 24
    assert derived.patience == 24
    assert derived.untouched == "ok"
