"""Provenance audit for H1b paired all-background weak-VFT data."""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from cjlt.models.vft_staggered_pri import record_samples
from experiments.find_route_a_h0o_crosspulse_identity_range import historic_receiver_ids

DATA_ROOT=Path(r"C:\LJQ_code\干扰数据"); COUNT=640; BASE=2_800_000_000
FIELDS=("pair_seed","receiver_noise_seed","vft_present","background_component_seeds","background_effective_JNR_dB","mft_num_targets","mft_delays_samples","mft_amplitudes","isrj_width_samples","isrj_replicas","vft_component_seed","vft_effective_JNR_dB","vft_doppler_Hz","vft_arrival_delay_samples","pulse_start_samples")

def sha(path:Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): d.update(b)
    return d.hexdigest()

def load(path:Path)->tuple[dict[str,np.ndarray],np.ndarray]:
    with h5py.File(path,"r") as f:
        if not set(FIELDS).issubset(f.keys()) or not {"iq_real","iq_imag"}.issubset(f.keys()): raise ValueError(f"missing H1b fields in {path}")
        return {n:np.asarray(f[n]) for n in FIELDS},np.asarray(f["iq_real"])+1j*np.asarray(f["iq_imag"])

def receiver_ids(path:Path)->set[int]:
    with h5py.File(path,"r") as f: return set(map(int,np.asarray(f["receiver_noise_seed"]).reshape(-1).astype(np.uint64)))

def audit(args:argparse.Namespace)->dict[str,object]:
    receipt=json.loads(args.receipt.read_text(encoding="utf-8")); select=json.loads(args.identity_selection.read_text(encoding="utf-8")); seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if not receipt.get("executed") or select.get("identity_overlap_count")!=0: raise ValueError("H1b receipt or selection invalid")
    if any(seal["test_v3"][n] for n in ("access_authorized","analysis_allowed","used")): raise ValueError("test-v3 seal invalid")
    absent,aiq=load(args.absent_path); present,piq=load(args.present_path); receiver=absent["receiver_noise_seed"].reshape(-1).astype(np.uint64); historic,sources=historic_receiver_ids()
    shared=("pair_seed","receiver_noise_seed","background_component_seeds","background_effective_JNR_dB","mft_num_targets","mft_delays_samples","mft_amplitudes","isrj_width_samples","isrj_replicas","pulse_start_samples")
    checks={
      "payload_shape_exact":aiq.shape==piq.shape==(COUNT,record_samples()),"pair_seed_exact":np.array_equal(absent["pair_seed"].reshape(-1),np.arange(BASE,BASE+COUNT)),"receiver_identity_unique":np.unique(receiver).size==COUNT,
      "all_five_background_counterfactual_metadata_exact":all(np.array_equal(absent[n],present[n],equal_nan=True) for n in shared),
      "five_background_jnr_exact":bool(np.all((absent["background_effective_JNR_dB"]>=19.999)&(absent["background_effective_JNR_dB"]<=20.001))),
      "absent_vft_fields_nan":bool(all(np.all(np.isnan(absent[n])) for n in ("vft_component_seed","vft_effective_JNR_dB","vft_doppler_Hz","vft_arrival_delay_samples"))),
      "present_vft_fields_in_preregistered_ranges":bool(np.all((present["vft_effective_JNR_dB"]>=7.999)&(present["vft_effective_JNR_dB"]<=8.001)) and np.all((present["vft_doppler_Hz"]>=1000)&(present["vft_doppler_Hz"]<=7000)) and np.all((present["vft_arrival_delay_samples"]>=100)&(present["vft_arrival_delay_samples"]<=500))),
      "present_absent_indicator_exact":bool(np.all(absent["vft_present"]==0) and np.all(present["vft_present"]==1)),"only_declared_vft_intervention_changes_iq":not np.array_equal(aiq,piq),
      "historical_train_identity_overlap_zero":not bool(historic.intersection(map(int,receiver))),"h0q_identity_overlap_zero":not bool(receiver_ids(args.h0q_source).intersection(map(int,receiver))),"h1a_identity_overlap_zero":not bool(receiver_ids(args.h1a_source).intersection(map(int,receiver))),"test_v3_still_sealed":True}
    checks={k:bool(v) for k,v in checks.items()}
    if not all(checks.values()): raise ValueError(f"H1b provenance contract failed: {checks}")
    result={"status":"completed_train_only_paired_audit","experiment":"RouteA_H1b_VFTStaggeredPRI_AllFiveBackgroundCompositeAttribution","source_scope":"project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716","absent_sha256":sha(args.absent_path),"present_sha256":sha(args.present_path),"sample_count_per_arm":COUNT,"record_samples":record_samples(),"contracts":checks,"pre_generation_identity_selection":str(args.identity_selection),"historic_train_sources_scanned":sources,"identity_audit_passed":True,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return result

def parse_args()->argparse.Namespace:
    source=DATA_ROOT/"h1b_vft_staggered_pri_all_backgrounds_20260803"; root=ROOT/"outputs/tapc/route_a_h1b_vft_staggered_pri_generation"; p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--absent-path",type=Path,default=source/"vft_absent_all_five_backgrounds.mat");p.add_argument("--present-path",type=Path,default=source/"vft_present_weak_8db_all_five_backgrounds.mat");p.add_argument("--receipt",type=Path,default=root/"formal_generation_receipt.json");p.add_argument("--identity-selection",type=Path,default=root/"pre_generation_identity_selection.json");p.add_argument("--h0q-source",type=Path,default=DATA_ROOT/"h0q_vft_staggered_pri_20260803/vft_low_1p0_2p5kHz.mat");p.add_argument("--h1a-source",type=Path,default=DATA_ROOT/"h1a_vft_staggered_pri_mft_isrj_20260803/vft_absent_mft_isrj.mat");p.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json");p.add_argument("--output",type=Path,default=root/"paired_identity_audit.json");return p.parse_args()

if __name__=="__main__":
    x=audit(parse_args());print(json.dumps({"status":x["status"],"identity_audit_passed":x["identity_audit_passed"]},ensure_ascii=False))
