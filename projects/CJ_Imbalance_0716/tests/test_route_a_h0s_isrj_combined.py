from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_h0s_generator_and_ar1_materializer_are_fixed_to_preregistered_shift():
    generator = (ROOT / 'experiments/generate_route_a_h0s_isrj_combined_base.m').read_text(encoding='utf8')
    transform = (ROOT / 'experiments/materialize_route_a_h0s_ar1_iq.m').read_text(encoding='utf8')
    assert 'cfg.seed = 810000000;' in generator
    assert 'cfg.params.ISRJ.sample_width_s = [25e-6,30e-6];' in generator
    assert 'cfg.params.ISRJ.num_replicas = [9,9];' in generator
    assert "'train','error'" in generator
    assert 'rho = 0.80;' in transform
    assert 'base_iq - exact_reference_white(seed) + ar1(seed)' in transform
    assert 'test_v3_used' in generator and 'test_v3_used' in transform


def test_h0s_python_defaults_are_train_only_and_frozen():
    from experiments import audit_route_a_h0s_isrj_combined as audit
    from experiments import run_route_a_h0s_isrj_combined_frozen as runner

    parser = runner.build_parser()
    args = parser.parse_args([])
    assert '0803_isrj_ood_h0s_r1_ar1' in str(args.data_path)
    assert args.experiment == 'RouteA_H0s_ISRJCombinedParameterNoiseFrozen'
    assert args.seeds == [2039, 2040, 2041, 2042, 2043]
    assert audit.RECEIVER_RANGE == (1008677743, 1008682782)
    assert audit.COUNT == 5040
