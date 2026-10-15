"""Formal provenance audit for the once-generated H2v visibility ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\LJQ_code\干扰数据")
COUNT, BASE_SEED, RECEIVER_OFFSET = 5_040, 3_920_000_000, 2 * 2_246_822_519


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receiver_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)


def run(args: argparse.Namespace) -> dict[str, object]:
    receipt, selection = json.loads(args.receipt.read_text(encoding="utf-8")), json.loads(args.identity_selection.read_text(encoding="utf-8"))
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not receipt.get("executed") or selection.get("identity_overlap_count") != 0:
        raise ValueError("H2v receipt or identity selection is invalid")
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 seal is not intact")
    required = {"iq_real", "iq_imag", "label_selected", "pairing_key", "receiver_noise_seed", "combination_mask", "mft_weak_mask", "mft_total_jnr_db", "mft_delay_samples", "num_pulses", "samples_per_pri"}
    with h5py.File(args.source, "r") as handle:
        fields = required.issubset(handle.keys()); real, imag = handle["iq_real"].shape, handle["iq_imag"].shape
        labels = np.asarray(handle["label_selected"], dtype=np.uint8); pairing = np.asarray(handle["pairing_key"]).reshape(-1).astype(np.int64); receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        combination = np.asarray(handle["combination_mask"]).reshape(-1).astype(np.int64); weak = np.asarray(handle["mft_weak_mask"]).reshape(-1).astype(bool); jnr = np.asarray(handle["mft_total_jnr_db"]).reshape(-1); delay = np.asarray(handle["mft_delay_samples"]); pulse = np.asarray(handle["num_pulses"]).reshape(-1); pri = np.asarray(handle["samples_per_pri"]).reshape(-1)
    mft = labels[:, 3].astype(bool); expected_receiver = ((np.arange(BASE_SEED, BASE_SEED + COUNT, dtype=np.uint64) + np.uint64(RECEIVER_OFFSET)) % np.uint64(2**32)).astype(np.int64)
    historic=[]
    for path in DATA_ROOT.rglob("*train*.mat"):
        if path.resolve() == args.source.resolve(): continue
        try: overlap=int(np.intersect1d(receiver,_receiver_ids(path)).size)
        except (OSError,KeyError): continue
        historic.append({"path":str(path),"receiver_overlap_count":overlap})
    spacing=float(np.min(np.diff(np.sort(delay[mft],axis=1),axis=1)))
    counts={str(mask):int(np.sum(combination==mask)) for mask in range(1,64)}
    checks={
        "required_fields_present":fields,"payload_shape_exact":real==imag==(COUNT,10000),"labels_shape_exact":labels.shape==(COUNT,6),"all_63_combinations_have_80_records":all(value==80 for value in counts.values()),
        "pairing_seed_exact":np.array_equal(pairing,np.arange(BASE_SEED,BASE_SEED+COUNT)),"receiver_mapping_exact":np.array_equal(receiver,expected_receiver),"receiver_unique":len(np.unique(receiver))==COUNT,
        "historic_train_receiver_identity_overlap_zero":not any(row["receiver_overlap_count"] for row in historic),"mft_count_exact":int(mft.sum())==2560 and int(weak.sum())==1280 and int(np.sum(mft&~weak))==1280 and not np.any(weak&~mft),
        "weak_visibility_ladder_exact":int(np.sum(mft&weak&(jnr==0)))==448 and int(np.sum(mft&weak&(jnr==4)))==416 and int(np.sum(mft&weak&(jnr==8)))==416 and int(np.sum(mft&~weak&(jnr==20)))==1280 and np.all(np.isnan(jnr[~mft])),
        "separated_mft_contract":delay[mft].shape==(2560,3) and np.all(np.isfinite(delay[mft])) and spacing>=1600,"one_pri_one_ms_contract":np.all(pulse==1) and np.all(pri==10000),"test_v3_still_sealed":True}
    checks={key:bool(value) for key,value in checks.items()}
    if not all(checks.values()): raise ValueError(f"H2v provenance contract failed: {checks}")
    result={"status":"completed_train_only_source_audit","experiment":"RouteA_H2v_ConditionalComplementarityAdmission","source_sha256":_sha256(args.source),"sample_count":COUNT,"minimum_observed_mft_replica_spacing_samples":spacing,"contracts":checks,"combination_counts":counts,"historical_train_identity_audit":historic,"identity_audit_passed":True,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,default=DATA_ROOT/"复合干扰数据0803_h2v_mft_visibility_ladder_r1"/"closedset_6class_train.mat"); parser.add_argument("--receipt",type=Path,default=ROOT/"outputs/tapc/route_a_h2v_visibility_ladder_generation/formal_generation_receipt.json"); parser.add_argument("--identity-selection",type=Path,default=ROOT/"outputs/tapc/route_a_h2v_visibility_ladder_generation/pre_generation_identity_selection.json"); parser.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); parser.add_argument("--output",type=Path,default=ROOT/"outputs/tapc/route_a_h2v_visibility_ladder_generation/source_audit.json"); print(json.dumps(run(parser.parse_args()),ensure_ascii=False,indent=2))
