import numpy as np

from cj0724.p3b_isrj_regime_oracle_sufficiency import load_saved_global


def test_load_saved_global_round_trip(tmp_path):
    run_dir = tmp_path / "seed_2041" / "A0_acf_ordinal" / "ID_control"
    run_dir.mkdir(parents=True)
    np.savez_compressed(
        run_dir / "predictions.npz",
        identity=np.asarray([1, 2]), prediction=np.asarray([1.0, 2.0]),
    )
    loaded = load_saved_global(tmp_path, ".", "ID_control")
    assert np.array_equal(loaded["identity"], np.asarray([1, 2]))
    assert np.array_equal(loaded["prediction"], np.asarray([1.0, 2.0]))
