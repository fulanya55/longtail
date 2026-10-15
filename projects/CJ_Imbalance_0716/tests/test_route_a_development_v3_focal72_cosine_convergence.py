from experiments.run_route_a_development_v3_focal72_cosine_convergence import (
    FOCAL36, H1, VARIANT, build_parser,
)


def test_fixed_focal72_protocol_defaults():
    args = build_parser().parse_args([])
    assert VARIANT == 'E5_Focal_72_Cosine'
    assert FOCAL36 == 'Focal36_Reference'
    assert H1 == 'H1_FrozenLinearBCE_Reference'
    assert args.epochs == args.patience == 72
    assert args.lr_schedule == 'cosine'
    assert args.minimum_lr_ratio == .01
