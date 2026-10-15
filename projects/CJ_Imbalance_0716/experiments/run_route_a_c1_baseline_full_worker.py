"""ASCII-launchable worker for the full C1 baseline (keeps Unicode paths in source)."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"E:\LJQ_code\CJ_Imbalance_0716")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.run_route_a_c1_baseline import run, parse_args  # noqa: E402

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--data-root", r"E:\论文相关\干扰数据\复合干扰数据0716\九类",
        "--f2-checkpoint-root", r"E:\LJQ_code\CJ_Imbalance_0716\frozen\20260731_route_a_visibility_balanced_fusion_v2",
        "--output-root", r"E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_c1_baseline_full_20260826",
        "--seed", "2039", "--split-seed", "260826", "--train-power", "30", "--primary-test-power", "30",
        "--val-fraction", "0.2", "--max-fit-samples", "0", "--max-primary-test-samples", "0",
        "--diagnostic-samples", "2048", "--batch-size", "32", "--eval-batch-size", "64", "--epochs", "8",
        "--lr", "1e-4", "--weight-decay", "1e-4", "--device", "cuda:0",
    ]
    run(parse_args())
