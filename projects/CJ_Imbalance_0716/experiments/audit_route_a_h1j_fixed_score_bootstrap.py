"""Read-only paired identity-bootstrap audit of the fixed H1j VFT scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.vft_dechirped_doppler import (  # noqa: E402
    dechirped_doppler_hz, fixed_phase_permutation, lfm_reference_phase,
)
from experiments.run_route_a_development_v2_h5 import _component_power, _sha256  # noqa: E402
from experiments.run_route_a_h1j_vft_dechirped_composite import _matched_view  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402


VFT = 4


def _source() -> Path:
    found = sorted(Path(r"C:\LJQ_code\干扰数据").glob("*0802_h1j_vft_dechirped_composite"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one H1j source, found {found}")
    return found[0] / "closedset_6class_train.mat"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _score(signals: np.ndarray, reference: torch.Tensor, device: torch.device, batch: int) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), batch):
            rows.append(dechirped_doppler_hz(torch.from_numpy(signals[start:start + batch]).to(device), reference).cpu().numpy())
    return np.concatenate(rows).astype(np.float64)


def _probability(score: np.ndarray, head: dict) -> np.ndarray:
    raw = ((score - float(head["mean"])) / float(head["std"])) * float(head["weight"]) + float(head["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))


def _rank(y: np.ndarray, score: np.ndarray) -> dict:
    return {"count": int(len(y)), "auroc": float(roc_auc_score(y, score)), "aupr": float(average_precision_score(y, score))}


def _bootstrap(record: dict, seed: int) -> dict:
    """10,000 paired receiver-identity draws for one unique frozen ranking.

    H1j's five optimizer seeds retain exactly the same candidate/control ranks,
    so resampling them would create pseudo-replication.  This CI therefore
    measures only identity uncertainty, which is the defensible quantity here.
    """
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in ("weak_auroc", "weak_aupr", "strong_auroc")}
    for _ in range(10_000):
        for endpoint, key, metric in (("weak", "weak_auroc", "auroc"), ("weak", "weak_aupr", "aupr"), ("strong", "strong_auroc", "auroc")):
            source = record[endpoint]
            for _retry in range(32):
                ids = rng.integers(0, len(source["y"]), size=len(source["y"]))
                y = source["y"][ids]
                if np.unique(y).size == 2:
                    deltas[key].append(_rank(y, source["candidate"][ids])[metric] - _rank(y, source["control"][ids])[metric])
                    break
            else:
                raise RuntimeError("bootstrap lost a class repeatedly")
    summary = {}
    for name, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {"bootstrap_95_ci": [float(x) for x in np.percentile(array, [2.5, 97.5])], "bootstrap_mean": float(array.mean()), "bootstrap": "paired_identity_10000"}
    return summary


def run(args: argparse.Namespace) -> dict:
    audit, seal = _read(args.audit), _read(args.test_v3_manifest)
    if not audit.get("identity_audit_passed") or any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("H1j provenance or test-v3 seal failed")
    if _sha256(args.data_path).lower() != audit["sha256"].lower():
        raise ValueError("H1j source hash differs from audited source")
    labels = read_labels(args.data_path); params, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(params), 10.0)
    roles = np.load(args.roles, allow_pickle=False); evaluation = np.asarray(roles["evaluation"], dtype=np.int64)
    signals = _load_signals(args.data_path, np.arange(len(labels)), args.io_batch_size)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = lfm_reference_phase(1000, device=device)
    correct = _score(signals, reference, device, args.score_batch_size)
    permuted = _score(signals, fixed_phase_permutation(reference), device, args.score_batch_size)
    weak_keep, weak_y, weak_counts = _matched_view(labels[evaluation], weak[evaluation], evaluation=True)
    strong_keep, strong_y, strong_counts = _matched_view(labels[evaluation], ~weak[evaluation], evaluation=True)
    records, per_seed = [], []
    for seed_dir in sorted(args.seed_root.glob("seed_*")):
        item = _read(seed_dir / "seed_summary.json")
        arms = item["arms"]
        candidate_weak = _probability(correct[evaluation][weak_keep], arms["CandidateCorrectDechirp"]["head"])
        control_weak = _probability(permuted[evaluation][weak_keep], arms["ControlPermutedPhase"]["head"])
        candidate_strong = _probability(correct[evaluation][strong_keep], arms["CandidateCorrectDechirp"]["head"])
        control_strong = _probability(permuted[evaluation][strong_keep], arms["ControlPermutedPhase"]["head"])
        values = {
            "seed": int(item["seed"]),
            "weak": {"candidate": _rank(weak_y, candidate_weak), "control": _rank(weak_y, control_weak)},
            "strong": {"candidate": _rank(strong_y, candidate_strong), "control": _rank(strong_y, control_strong)},
            "head_weights": {name: float(arms[name]["head"]["weight"]) for name in arms},
        }
        values["delta"] = {
            "weak_auroc": values["weak"]["candidate"]["auroc"] - values["weak"]["control"]["auroc"],
            "weak_aupr": values["weak"]["candidate"]["aupr"] - values["weak"]["control"]["aupr"],
            "strong_auroc": values["strong"]["candidate"]["auroc"] - values["strong"]["control"]["auroc"],
        }
        per_seed.append(values)
        records.append({"weak": {"y": weak_y, "candidate": candidate_weak, "control": control_weak}, "strong": {"y": strong_y, "candidate": candidate_strong, "control": control_strong}})
    if len(records) != 5 or any(value["head_weights"]["CandidateCorrectDechirp"] <= 0 or value["head_weights"]["ControlPermutedPhase"] <= 0 for value in per_seed):
        raise ValueError("unexpected H1j seed/head contract")
    ranking_equal = all(
        np.array_equal(np.argsort(records[0]["weak"]["candidate"]), np.argsort(row["weak"]["candidate"]))
        and np.array_equal(np.argsort(records[0]["weak"]["control"]), np.argsort(row["weak"]["control"]))
        for row in records[1:]
    )
    if not ranking_equal:
        raise ValueError("unexpected seed-varying ranking; this fixed-score audit needs a new preregistered statistic")
    boot = _bootstrap(records[0], 202608021)
    observed = {name: float(np.mean([row["delta"][name] for row in per_seed])) for name in boot}
    checks = {
        "weak_auroc_mean_ge_0p015": observed["weak_auroc"] >= 0.015,
        "weak_auroc_identity_bootstrap_lower_gt_zero": boot["weak_auroc"]["bootstrap_95_ci"][0] > 0.0,
        "weak_aupr_identity_bootstrap_lower_gt_zero": boot["weak_aupr"]["bootstrap_95_ci"][0] > 0.0,
        "strong_auroc_identity_bootstrap_lower_ge_minus_0p010": boot["strong_auroc"]["bootstrap_95_ci"][0] >= -0.010,
        "five_seed_rankings_not_independent": ranking_equal,
    }
    result = {
        "status": "completed_read_only_fixed_score_audit", "experiment": "RouteA_H1j_FixedScorePairedIdentityBootstrap",
        "evidence_role": "read_only_statistical_reaudit_of_existing_H1j_source", "data_sha256": _sha256(args.data_path),
        "source_audit_sha256": _sha256(args.audit), "matched_counts": {"weak": weak_counts, "strong": strong_counts},
        "per_seed": per_seed, "observed_mean_delta": observed, "paired_identity_bootstrap": boot, "checks": checks,
        "decision": "H1j_fixed_score_supports_single_source_VFT_mechanism_only" if all(value for key, value in checks.items() if key != "five_seed_rankings_not_independent") else "H1j_fixed_score_does_not_support_VFT_mechanism",
        "limitations": ["five optimizer seeds preserve the same rankings and are not independent ranking replications", "no new training, calibration, threshold or fusion was performed", "does not authorize H2 or test-v3"],
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"decision": result["decision"], "observed": observed, "bootstrap": boot, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=_source())
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/identity_audit.json")
    parser.add_argument("--roles", type=Path, default=ROOT / "outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/role_indices.npz")
    parser.add_argument("--seed-root", type=Path, default=ROOT / "outputs/tapc/route_a_h1j_vft_dechirped_composite")
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tapc/route_a_h1j_fixed_score_bootstrap/route_a_h1j_fixed_score_bootstrap_summary.json")
    parser.add_argument("--device", default=None); parser.add_argument("--io-batch-size", type=int, default=128); parser.add_argument("--score-batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
