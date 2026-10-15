import numpy as np

from cj0724.p2k_confirmation_aggregate import combined_bootstrap, identity_sufficient_statistics


def _arrays(candidate_shift=0.0):
    identities = np.repeat(np.asarray([10, 20], dtype=np.int64), 2)
    labels = np.ones((4, 6), dtype=np.uint8)
    mask = np.ones((4, 6, 2), dtype=np.uint8)
    target = np.zeros((4, 6, 2), dtype=np.float32)
    old = np.full_like(target, 0.2)
    candidate = old.copy()
    candidate[:, 1, 1] = 0.2 + candidate_shift
    return {
        "receiver_noise_seed": identities,
        "labels": labels,
        "mask": mask,
        "routed_probability": np.ones((4, 6), dtype=np.float32),
        "routed_thresholds": np.full(6, 0.5, dtype=np.float32),
        "target": target,
        "old_joint_prediction": old,
        "calibrated_joint_prediction": candidate,
        "parameter_scale": np.ones((6, 2), dtype=np.float32),
    }


def test_sufficient_statistics_and_bootstrap_detect_improvement():
    arrays = _arrays(candidate_shift=-0.1)
    group = {
        "candidate": identity_sufficient_statistics(arrays, "calibrated_joint_prediction"),
        "old": identity_sufficient_statistics(arrays, "old_joint_prediction"),
    }
    result = combined_bootstrap([group], "predicted_gate_parameter_nmae", 50, 7)
    assert result["observed_mean_delta"] < 0.0
    assert result["ci95_high"] < 0.0
