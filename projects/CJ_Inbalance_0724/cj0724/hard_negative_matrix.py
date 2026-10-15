"""Run E1/E2 VFT hard-negative experiments on the frozen D11 protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = {
    "e1_ohem": "ohem",
    "e2_ohem_rank": "ohem_rank",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--power-root",
        type=Path,
        default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vft_d11_hard_negative_20260725"),
    )
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = [(method, seed) for method in args.methods for seed in args.seeds]
    for index, (method, seed) in enumerate(jobs, start=1):
        run_dir = args.output_root / "VFT" / "D11" / method / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(
                f"SKIP {index}/{len(jobs)} {method} seed={seed} "
                f"target_AUPR={summary['primary_target_aupr']:.6f}"
            )
            continue
        command = [
            sys.executable,
            "-m",
            "cj0724.dual_axis_train",
            "--power-root",
            str(args.power_root),
            "--target-label",
            "VFT",
            "--cell",
            "D11",
            "--model",
            "efficientnet_b1_1d",
            "--loss",
            METHODS[method],
            "--view-mode",
            "single",
            "--method-name",
            method,
            "--ohem-weight",
            "0.5",
            "--ohem-negative-ratio",
            "2.0",
            "--ranking-weight",
            "0.2",
            "--ranking-margin",
            "0.5",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--output-root",
            str(args.output_root),
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(f"START {index}/{len(jobs)} {method} seed={seed}")
        subprocess.run(command, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"DONE {index}/{len(jobs)} {method} seed={seed} "
            f"macro_AUPR={summary['primary_macro_aupr']:.6f} "
            f"target_AUPR={summary['primary_target_aupr']:.6f}"
        )


if __name__ == "__main__":
    main()
