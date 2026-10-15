"""Preregistered continuous-visibility VTSE validation on development-v3."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_vtse_soft_strong_anchor_v3 import parse_args, run


if __name__ == "__main__":
    args = parse_args()
    args.training_mode = "continuous"
    default_soft = PROJECT_ROOT / "outputs" / "tapc" / "route_a_vtse_soft_strong_anchor_v3"
    if args.output_root == default_soft:
        args.output_root = PROJECT_ROOT / "outputs" / "tapc" / "route_a_vtse_continuous_visibility_v3"
    run(args)
