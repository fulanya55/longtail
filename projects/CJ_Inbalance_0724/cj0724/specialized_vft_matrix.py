"""Run E3 specialized bounded VFT residual experts on the frozen D11 protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    method = "e3_specialized_residual_moe"
    for index, seed in enumerate(args.seeds, start=1):
        run_dir = args.output_root / "VFT" / "D11" / method / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"SKIP {index}/{len(args.seeds)} seed={seed} target_AUPR={summary['primary_target_aupr']:.6f}")
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
            "efficientnet_b1_vft_residual_moe_1d",
            "--loss",
            "vft_residual_moe",
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
        print(f"START {index}/{len(args.seeds)} seed={seed}")
        subprocess.run(command, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"DONE {index}/{len(args.seeds)} seed={seed} "
            f"macro_AUPR={summary['primary_macro_aupr']:.6f} "
            f"target_AUPR={summary['primary_target_aupr']:.6f}"
        )


if __name__ == "__main__":
    main()
