"""Generate per-label and per-combination comparison tables from saved checkpoints."""

from __future__ import annotations

import argparse

from cj0724.evaluation import evaluate_benchmark_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_benchmark_run(args.run_dir, args.device, args.models, args.workers)
    print(f"Detailed evaluation complete: {result}")


if __name__ == "__main__":
    main()
