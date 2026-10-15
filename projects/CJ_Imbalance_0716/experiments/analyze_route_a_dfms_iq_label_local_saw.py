"""Post-hoc residual-logit diagnostics for the label/local/Saw structure screen."""

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_controlled_lt import _write  # noqa: E402
from experiments.run_route_a_dfms_iq_structure_screen import CLASS_NAMES  # noqa: E402


STEPS = (
    ("Q1_LabelQuery", "C0_H2"),
    ("Q2_LocalTopK", "Q1_LabelQuery"),
    ("Q3_EarlySawExpert", "Q2_LocalTopK"),
)


def _logits(root, frozen_head_root, seed, variant):
    if variant == "C0_H2":
        path = (
            frozen_head_root
            / "seed_{}".format(seed)
            / "H2_PosBCE"
            / "predictions.npz"
        )
    else:
        path = root / "seed_{}".format(seed) / variant / "predictions.npz"
    return np.load(path)["logits"].astype(np.float64)


def run(args):
    per_run = []
    for seed in args.seeds:
        cache = {
            variant: _logits(args.root, args.frozen_head_root, seed, variant)
            for variant in {item for pair in STEPS for item in pair}
        }
        for candidate, baseline in STEPS:
            residual = cache[candidate] - cache[baseline]
            per_run.append(
                {
                    "seed": int(seed),
                    "candidate": candidate,
                    "baseline": baseline,
                    "per_label_mean": {
                        name: float(residual[:, index].mean())
                        for index, name in enumerate(CLASS_NAMES)
                    },
                    "per_label_std": {
                        name: float(residual[:, index].std())
                        for index, name in enumerate(CLASS_NAMES)
                    },
                    "maximum_absolute_residual": float(np.abs(residual).max()),
                }
            )
    payload = {
        "status": "completed",
        "locked_test_used": False,
        "interpretation_guard": (
            "large nonzero residual means indicate class-prior/logit shifts, "
            "not automatically label-specific local evidence"
        ),
        "per_run": per_run,
    }
    _write(args.root / "residual_logit_diagnostics.json", payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_dfms_iq_label_local_saw",
    )
    parser.add_argument(
        "--frozen-head-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "tapc"
        / "route_a_dfms_iq_frozen_heads",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
