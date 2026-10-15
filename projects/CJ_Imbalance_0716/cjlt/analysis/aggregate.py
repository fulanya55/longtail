"""Seed-level aggregation and paired comparisons for formal result tables."""

import math

import numpy as np


def aggregate_metric(values):
    data = np.asarray(list(values), dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if len(data) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(float(len(data))) if len(data) > 1 else 0.0
    return {
        "n": int(len(data)),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def paired_difference(baseline_by_seed, method_by_seed):
    seeds = sorted(set(baseline_by_seed).intersection(method_by_seed))
    if not seeds:
        raise ValueError("paired comparison has no common seeds")
    differences = [
        float(method_by_seed[seed]) - float(baseline_by_seed[seed])
        for seed in seeds
    ]
    summary = aggregate_metric(differences)
    return {
        "seeds": seeds,
        "differences": differences,
        "mean_difference": summary["mean"],
        "std_difference": summary["std"],
        "ci95_low": summary["ci95_low"],
        "ci95_high": summary["ci95_high"],
    }

