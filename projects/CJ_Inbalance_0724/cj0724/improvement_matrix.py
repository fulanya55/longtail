"""Run the controlled VFT-D11 backbone/loss improvement matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = {
    "b1_bce": ("efficientnet_b1_1d", "bce"),
    "b1_asl": ("efficientnet_b1_1d", "asl"),
    "moe3_bce": ("efficientnet_b1_moe3_1d", "bce"),
    "moe3_asl": ("efficientnet_b1_moe3_1d", "asl"),
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
        default=Path("outputs/vft_d11_backbone_loss_20260725"),
    )
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = [(method, seed) for method in args.methods for seed in args.seeds]
    for job_index, (method, seed) in enumerate(jobs, start=1):
        model, loss = METHODS[method]
        run_dir = args.output_root / "VFT" / "D11" / method / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(
                f"SKIP {job_index}/{len(jobs)} {method} seed={seed} "
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
            model,
            "--loss",
            loss,
            "--view-mode",
            "single",
            "--method-name",
            method,
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
        print(f"START {job_index}/{len(jobs)} {method} seed={seed}")
        subprocess.run(command, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"DONE {job_index}/{len(jobs)} {method} seed={seed} "
            f"macro_AUPR={summary['primary_macro_aupr']:.6f} "
            f"target_AUPR={summary['primary_target_aupr']:.6f}"
        )


if __name__ == "__main__":
    main()
