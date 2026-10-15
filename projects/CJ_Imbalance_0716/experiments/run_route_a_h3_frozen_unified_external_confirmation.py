"""One-shot, no-refit H3 external confirmation of the frozen unified candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import extract_scores
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import PEAK_INDEX, _load_json, _run_seed
from experiments.run_route_a_h1_b2_mft_vft_development_v12_frozen import SHARED_VARIANT, _mft_gate, _promote_shared_rows, _shared_variant
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft, _goal_rows, _vft_gate
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config
from experiments.run_shared import _data_path, load_config

SOURCE = Path(r"C:\LJQ_code\干扰数据\复合干扰数据0803_h3_frozen_unified_confirmation_r1\closedset_6class_train.mat")
SEEDS = (2039, 2040, 2041, 2042, 2043)


def _verify_record(record: dict[str, object]) -> None:
    path = Path(str(record["path"]))
    if not path.is_file() or _sha256(path).lower() != str(record["sha256"]).lower():
        raise ValueError(f"frozen artifact mismatch: {path}")


def _verify_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("candidate_name") != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("unexpected candidate manifest")
    for seed in SEEDS:
        for value in manifest["seed_artifacts"][str(seed)].values():
            if isinstance(value, dict) and "sha256" in value:
                _verify_record(value)
    for value in manifest["source_code"].values():
        _verify_record(value)


def smoke(args: argparse.Namespace) -> dict[str, object]:
    """Artifact-only smoke; never opens the H3 source."""
    seal, manifest = _load_json(args.test_v3_manifest), _load_json(args.frozen_manifest)
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    _verify_manifest(manifest)
    result = {"status": "completed_artifact_only_smoke", "formal_h3_source_opened": False, "candidate_name": manifest["candidate_name"], "seed_count": len(SEEDS), "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "smoke.json", result); return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.smoke:
        return smoke(args)
    audit, seal, manifest = _load_json(args.audit), _load_json(args.test_v3_manifest), _load_json(args.frozen_manifest)
    if not audit.get("identity_audit_passed") or audit.get("evidence_role") != args.expected_evidence_role or audit.get("test_files_opened"):
        raise ValueError("frozen-confirmation source audit is invalid")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    _verify_manifest(manifest)
    if _sha256(args.source).lower() != audit["sha256"].lower():
        raise ValueError("H3 source hash drift")
    config = load_config(args.config); _validate_config(config)
    labels3 = read_labels(_data_path(config)); params3, _ = read_parameters(_data_path(config))
    weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
    labels = read_labels(args.source); params, _ = read_parameters(args.source)
    weak = compute_weak_component_mask(labels, _component_power(params), args.weak_threshold_db)
    signals = _load_signals(args.source, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(args.source, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    periodic = _extract_periodicity(signals, device, args.periodicity_batch_size)
    vft = _extract_vft(signals, device, args.vft_batch_size)
    mft = extract_scores(signals, device, args.mft_batch_size)["MF_AlignedLFM"][:, PEAK_INDEX]
    rows=[]
    for seed in SEEDS:
        row=_run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic, vft, mft, seed, device, manifest)
        rows.append(_shared_variant(args, seed, row))
    vft_gate, mft_gate, unified_gate = _vft_gate(rows), _mft_gate(rows), _goal_rows(_promote_shared_rows(rows))
    result={"status":"completed_frozen_external_confirmation","experiment":args.experiment,"candidate_name":manifest["candidate_name"],"evidence_role":args.evidence_role,"data_sha256":audit["sha256"],"frozen_manifest_sha256":_sha256(args.frozen_manifest),"rows":rows,"vft_mechanism_gate":vft_gate,"mft_shared_policy_gate":mft_gate,"unified_promotion_gate":unified_gate,"confirmation_passed":bool(vft_gate["passed"] and mft_gate["passed"] and unified_gate["passed"]),"external_refit":False,"hyperparameter_search":False,"calibration_refit":False,"threshold_refit":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True); _write_json(args.output_root/"summary.json",result); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,default=SOURCE); parser.add_argument("--audit",type=Path,default=ROOT/"outputs/tapc/route_a_h3_frozen_confirmation_generation/identity_audit.json"); parser.add_argument("--frozen-manifest",type=Path,default=ROOT/"frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json"); parser.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); parser.add_argument("--config",type=Path,default=ROOT/"configs/student_a/six_class_development_v3.yaml"); parser.add_argument("--f2-root",type=Path,default=ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen"); parser.add_argument("--c6-root",type=Path,default=ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); parser.add_argument("--head-root",type=Path,default=ROOT/"outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines"); parser.add_argument("--unified-root",type=Path,default=ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); parser.add_argument("--periodicity-root",type=Path,default=ROOT/"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3"); parser.add_argument("--construction-root",type=Path,default=ROOT/"outputs/tapc/route_a_h1_b2_mft_vft_unified_v3"); parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/tapc/route_a_h3_frozen_unified_confirmation"); parser.add_argument("--experiment",default="RouteA_H3_FrozenUnifiedExternalConfirmation"); parser.add_argument("--evidence-role",default="fresh_standard_train_only_external_confirmation"); parser.add_argument("--expected-evidence-role",default="fresh_standard_train_only_external_confirmation"); parser.add_argument("--device",default=None); parser.add_argument("--io-batch-size",type=int,default=128); parser.add_argument("--backbone-batch-size",type=int,default=128); parser.add_argument("--eval-batch-size",type=int,default=128); parser.add_argument("--raw-batch-size",type=int,default=64); parser.add_argument("--periodicity-batch-size",type=int,default=64); parser.add_argument("--vft-batch-size",type=int,default=32); parser.add_argument("--mft-batch-size",type=int,default=64); parser.add_argument("--raw-hidden-channels",type=int,default=16); parser.add_argument("--raw-topk",type=int,default=16); parser.add_argument("--raw-dropout",type=float,default=.1); parser.add_argument("--weak-threshold-db",type=float,default=10.); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args(); print(json.dumps(run(args),ensure_ascii=False,indent=2))
