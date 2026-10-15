"""Post-hoc, preregistered-gate analysis for the AMA development-v2 method sweep."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_ama_development_v2_20260903 import SRC_A, SRC_B, OUT as V2_OUT, load_selected
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import sha256

OUT = ROOT / "outputs" / "route_a_ama_longtail_methods_devv2_20260903"
SUMMARY = OUT / "summary.json"
CLASSES = ("FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ")
METRICS = ("macro_aupr", "macro_f1", "combination_exact_accuracy", "hamming_loss")


def codes(y: np.ndarray) -> np.ndarray:
    return y.astype(np.uint8) @ (1 << np.arange(6, dtype=np.uint8))


def checkpoint(method: str, seed: int) -> Path:
    if method == "bce":
        return V2_OUT / "if3" / f"if3__seed_{seed}.pth"
    return OUT / method / f"{method}__seed_{seed}.pth"


def infer(ck: Path, x: np.ndarray, device: torch.device) -> np.ndarray:
    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2).to(device)
    payload = torch.load(str(ck), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=128, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def bootstrap_ci(values: np.ndarray, seed: int = 20260903) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def exact_sign_permutation_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    perm = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        perm.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(perm) >= observed - 1e-15))


def paired_stats(values: np.ndarray) -> dict:
    try:
        wp = float(wilcoxon(values, alternative="two-sided", method="exact").pvalue)
    except ValueError:
        wp = 1.0
    return {
        "mean_delta": float(values.mean()),
        "std_delta": float(values.std(ddof=1)),
        "bootstrap_95ci_over_five_seeds": bootstrap_ci(values),
        "exact_sign_permutation_p_two_sided": exact_sign_permutation_p(values),
        "wilcoxon_p_two_sided": wp,
        "positive_direction_seeds": int((values > 0).sum()),
        "n_seed_pairs": int(len(values)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest = d["manifest"]
    i_sig, i_lab, _ = load_selected(SRC_B, manifest["if3_indices"])
    v_sig, v_lab, _ = load_selected(SRC_A, manifest["validation_indices"])
    t_sig, t_lab, _ = load_selected(SRC_A, manifest["development_test_indices"])
    fit_counts = i_lab.astype(np.uint8).sum(axis=0).astype(int)
    tail_index = int(np.argmin(fit_counts))
    head_index = int(np.argmax(fit_counts))
    combo_counts = Counter(int(x) for x in codes(i_lab))
    rare_n = int(np.ceil(0.20 * len(combo_counts)))
    rare_codes = [k for k, _ in sorted(combo_counts.items(), key=lambda kv: (kv[1], kv[0]))[:rare_n]]

    rows = defaultdict(dict)
    device = torch.device("cuda:0")
    for run in d["runs"]:
        method, seed = run["method"], int(run["seed"])
        ck = checkpoint(method, seed)
        entry = {
            "method": method,
            "seed": seed,
            "checkpoint": str(ck.resolve()),
            "checkpoint_sha256": sha256(ck),
            "elapsed_seconds": run.get("elapsed_seconds"),
        }
        for split, x, y in (("validation", v_sig, v_lab), ("development_test", t_sig, t_lab)):
            logits = infer(ck, x, device)
            pred = (logits >= 0.0).astype(np.uint8)
            yc, pc = codes(y), codes(pred)
            rare_mask = np.isin(yc, rare_codes)
            tail_pos = y[:, tail_index] == 1
            head_neg = y[:, head_index] == 0
            entry[split] = {
                **{k: float(run[split][k]) for k in METRICS},
                "tail_recall": float(pred[tail_pos, tail_index].mean()),
                "head_fpr": float(pred[head_neg, head_index].mean()),
                "rare_combination_accuracy": float((pc[rare_mask] == yc[rare_mask]).mean()),
                "rare_sample_count": int(rare_mask.sum()),
            }
        rows[method][seed] = entry

    methods = list(rows)
    baseline = rows["bce"]
    aggregate = {}
    gates = {}
    for method in methods:
        aggregate[method] = {}
        for split in ("validation", "development_test"):
            aggregate[method][split] = {}
            for metric in (*METRICS, "tail_recall", "head_fpr", "rare_combination_accuracy"):
                vals = np.asarray([rows[method][s][split][metric] for s in sorted(rows[method])])
                aggregate[method][split][metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)),
                }
        times = [rows[method][s]["elapsed_seconds"] for s in sorted(rows[method])]
        times = [float(x) for x in times if x is not None]
        aggregate[method]["training_elapsed_seconds"] = {
            "mean": float(np.mean(times)) if times else None,
            "std": float(np.std(times, ddof=1)) if len(times) > 1 else None,
            "n": len(times),
        }
        if method == "bce":
            continue
        split = "development_test"
        delta = {}
        for metric in (*METRICS, "tail_recall", "head_fpr", "rare_combination_accuracy"):
            vals = np.asarray([rows[method][s][split][metric] - baseline[s][split][metric] for s in sorted(rows[method])])
            delta[metric] = paired_stats(vals)
        checks = {
            "tail_recall_delta_ge_0.03": delta["tail_recall"]["mean_delta"] >= 0.03,
            "rare_combo_delta_ge_0.02": delta["rare_combination_accuracy"]["mean_delta"] >= 0.02,
            "macro_f1_non_decrease": delta["macro_f1"]["mean_delta"] >= 0.0,
            "macro_aupr_delta_ge_minus_0.001": delta["macro_aupr"]["mean_delta"] >= -0.001,
            "head_fpr_delta_le_0.005": delta["head_fpr"]["mean_delta"] <= 0.005,
            "overall_combo_non_decrease": delta["combination_exact_accuracy"]["mean_delta"] >= 0.0,
            "tail_positive_at_least_4_of_5": delta["tail_recall"]["positive_direction_seeds"] >= 4,
            "rare_positive_at_least_4_of_5": delta["rare_combination_accuracy"]["positive_direction_seeds"] >= 4,
        }
        gates[method] = {"development_test_delta_vs_bce": delta, "checks": checks, "promote": all(checks.values())}

    model = build_backbone("dfms_iq", num_classes=6, width=16, dropout=0.2)
    result = {
        "schema": "route-a-ama-longtail-methods-gate-analysis-v1",
        "status": "completed",
        "rare_definition": "bottom ceil(20%)=13 combinations by IF3 fit frequency; ties ordered by combination code",
        "rare_codes": rare_codes,
        "rare_fit_counts": {str(k): combo_counts[k] for k in rare_codes},
        "tail_label": CLASSES[tail_index],
        "head_label": CLASSES[head_index],
        "fit_label_counts": dict(zip(CLASSES, fit_counts.tolist())),
        "parameter_count_all_methods": int(sum(p.numel() for p in model.parameters())),
        "inference_architecture": "identical F2_DFMS_Full for all methods; no method adds inference modules",
        "rows": {m: {str(s): r for s, r in sr.items()} for m, sr in rows.items()},
        "aggregate": aggregate,
        "promotion_gates": gates,
        "statistical_limit": "n=5 seed pairs; exact two-sided sign/Wilcoxon p cannot be below 0.0625 when all five directions agree",
        "locked_test_used": False,
        "frozen_e5_modified": False,
        "evidence_boundary": "development-v2-confirmation_only",
    }
    target = OUT / "gate_analysis.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(target.resolve()), "promoted": [m for m, g in gates.items() if g["promote"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
