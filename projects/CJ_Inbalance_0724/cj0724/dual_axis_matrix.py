"""Sequential launcher for the EfficientNet-B1 dual-axis screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cj0724.dual_axis import CLASS_NAMES
from cj0724.dual_axis_train import CELLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", choices=CLASS_NAMES, default=["SawFM", "VFT"])
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loss", choices=("bce", "db"), default="bce")
    parser.add_argument("--view-mode", choices=("single", "pair"), default="single")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dual_axis_b1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = [
        (target, cell, seed)
        for target in args.targets
        for cell in args.cells
        for seed in args.seeds
    ]
    for job_index, (target, cell, seed) in enumerate(jobs, start=1):
        method = args.loss if args.view_mode == "single" else f"{args.loss}_pair"
        run_dir = args.output_root / target / cell / method / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            print(f"SKIP {job_index}/{len(jobs)} {target} {cell} seed={seed}")
            continue
        command = [
            sys.executable,
            "-u",
            "-m",
            "cj0724.dual_axis_train",
            "--target-label",
            target,
            "--cell",
            cell,
            "--loss",
            args.loss,
            "--view-mode",
            args.view_mode,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--output-root",
            str(args.output_root),
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(f"START {job_index}/{len(jobs)} {target} {cell} seed={seed}")
        subprocess.run(command, check=True)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            "DONE {}/{} {} {} seed={} macro_AUPR={:.6f} target_AUPR={:.6f}".format(
                job_index,
                len(jobs),
                target,
                cell,
                seed,
                payload["primary_macro_aupr"],
                payload["primary_target_aupr"],
            )
        )


if __name__ == "__main__":
    main()
