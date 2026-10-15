"""Generate the auditable pilot and formal experiment job registry."""

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.engine.run_manager import configuration_sha256  # noqa: E402


def _job(**kwargs):
    payload = dict(kwargs)
    payload["allow_locked_test"] = False
    payload["status"] = "planned"
    payload["config_hash"] = configuration_sha256(payload)
    return payload


def build_planned_jobs():
    jobs = []
    for variant in ("A0", "A2", "A3", "A4"):
        jobs.append(
            _job(
                branch="student_a",
                protocol="four_class_pilot",
                class_mode="four",
                model=variant,
                loss="cb_asl",
                seed=2026,
                labeled_ratio=1.0,
                imbalance_factor="balanced_pilot",
                power_setting="10_30_50" if variant in ("A3", "A4") else "base",
            )
        )
    for seed in (2026, 2027, 2028, 2029, 2030):
        for variant in ("A0", "A2", "A3", "A4"):
            jobs.append(
                _job(
                    branch="student_a",
                    protocol="six_class_formal_development",
                    class_mode="six",
                    model=variant,
                    loss="cb_asl",
                    seed=seed,
                    labeled_ratio=1.0,
                    imbalance_factor="label_combo_power_registered",
                    power_setting="9_levels" if variant in ("A3", "A4") else "base",
                )
            )
    for variant in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        jobs.append(
            _job(
                branch="student_b",
                protocol="four_class_20pct_pilot",
                class_mode="four",
                model=variant,
                loss="cb_asl",
                seed=2026,
                labeled_ratio=0.20,
                imbalance_factor="nested_label_budget",
                power_setting="same_distribution_unlabeled",
            )
        )
    for seed in (2026, 2027, 2028, 2029, 2030):
        for ratio in (0.05, 0.10, 0.20, 0.50):
            for variant in ("B0", "B5", "B6"):
                jobs.append(
                    _job(
                        branch="student_b",
                        protocol="six_class_low_label_formal_development",
                        class_mode="six",
                        model=variant,
                        loss="cb_asl",
                        seed=seed,
                        labeled_ratio=ratio,
                        imbalance_factor="nested_label_budget",
                        power_setting="same_and_cross_power_unlabeled",
                    )
                )
    return jobs


def write_planned_jobs(path, jobs):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(jobs[0].keys())
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(jobs)
    return target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "planned_jobs.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = write_planned_jobs(arguments.output, build_planned_jobs())
    print(str(output))

