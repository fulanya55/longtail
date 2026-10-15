"""Post-hoc development diagnostic of the frozen D1/D2 visibility heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_presence_visibility_aux_v3 import (  # noqa: E402
    AUX_VARIANT, NO_AUX_VARIANT, PresenceVisibilityHead, _head_scores,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES, TARGET_NAMES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256, _extract_backbone, _load_roles, _validate_config,
)
from experiments.run_route_a_visibility_gap_frozen_probe_v3 import _relative_power_gap  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


def _metrics(truth_db, prediction_normalized, weak_threshold_db, gap_clip_db):
    prediction_db = prediction_normalized * float(gap_clip_db)
    weak = truth_db <= -float(weak_threshold_db)
    return {
        "count": int(len(truth_db)), "weak_count": int(weak.sum()),
        "spearman": float(spearmanr(truth_db, prediction_db).statistic),
        "mae_db": float(mean_absolute_error(truth_db, prediction_db)),
        "weak_vs_strong_auroc": float(roc_auc_score(weak.astype(np.uint8), -prediction_db)),
    }


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    gap6 = _relative_power_gap(labels6, _component_power(values), -args.gap_clip_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for seed in args.seeds:
        roles, _ = _load_roles(args.c6_root, seed, data_path, config["normalization"], args.io_batch_size)
        state = torch.load(args.c6_root/f"seed_{seed}"/"C6_Joint"/"best_model.pth",
                           map_location="cpu", weights_only=False)["model_state_dict"]
        base = _load_model_from_state(args.f2_root/"runs"/F2_MODEL_NAME/f"seed_{seed}", device, state)
        extracted = _extract_backbone(base, {"early": roles["early"]}, device,
                                      args.feature_batch_size, args.topk)
        del base
        ids = roles["early"]["sample_ids"]
        seed_result = {"seed": int(seed), "variants": {}}
        for variant in (NO_AUX_VARIANT, AUX_VARIANT):
            checkpoint = torch.load(args.experiment_root/f"seed_{seed}"/f"{variant}.pth",
                                    map_location="cpu", weights_only=False)
            standardized = ((extracted["early"]["temporal_stats"] - checkpoint["mean"])
                            / np.maximum(checkpoint["std"], 1e-6)).astype(np.float32)
            head = PresenceVisibilityHead(standardized.shape[1], args.adapter_rank).to(device)
            head.load_state_dict(checkpoint["head_state_dict"], strict=True)
            _, prediction = _head_scores(head, standardized, device)
            label_metrics = {}
            for local, (index, name) in enumerate(zip(TARGET_INDICES, TARGET_NAMES)):
                valid = (labels6[ids, index] == 1) & np.isfinite(gap6[ids, index])
                label_metrics[name] = _metrics(gap6[ids[valid], index], prediction[valid, local],
                                               args.weak_threshold_db, args.gap_clip_db)
            seed_result["variants"][variant] = label_metrics
        rows.append(seed_result)
    aggregate = {}
    for variant_offset, variant in enumerate((NO_AUX_VARIANT, AUX_VARIANT)):
        aggregate[variant] = {}
        for label_offset, name in enumerate(TARGET_NAMES):
            aggregate[variant][name] = {
                metric: _mean_ci([row["variants"][variant][name][metric] for row in rows],
                                 20268100+variant_offset*1000+label_offset*10+metric_offset)
                for metric_offset, metric in enumerate(("spearman", "mae_db", "weak_vs_strong_auroc"))
            }
        aggregate[variant]["mean_across_labels"] = {
            metric: _mean_ci([
                np.mean([row["variants"][variant][name][metric] for name in TARGET_NAMES])
                for row in rows], 20268900+variant_offset*10+metric_offset)
            for metric_offset, metric in enumerate(("spearman", "weak_vs_strong_auroc"))
        }
    payload = {"status": "completed", "experiment": "RouteA_PresenceVisibilityAux_Diagnostic_v3",
               "data_sha256": data_sha, "seeds": list(args.seeds), "aggregate": aggregate,
               "per_seed": rows, "locked_test_used": False, "test_v2_used": False,
               "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root/"route_a_presence_visibility_aux_diagnostic_v3.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--experiment-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_presence_visibility_aux_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_presence_visibility_aux_v3_diagnostic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device", default=None); parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128); parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--adapter-rank", type=int, default=32); parser.add_argument("--gap-clip-db", type=float, default=40.0)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())

