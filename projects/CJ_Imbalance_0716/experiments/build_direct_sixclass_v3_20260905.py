"""Build a fresh six-class IF3 dataset from unused same-contract raw pools."""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.run_route_a_ama_development_v2_20260903 import SEL,EXC,pool_meta,distribution
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import DATA_ROOT,sha256

RAW_A_TEST=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1\closedset_9class_test.mat")
RAW_B_TRAIN=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\full_power_10_30dB_v1\closedset_9class_train.mat")
OUT=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA")
IF3_COUNTS=np.asarray(json.loads((DATA_ROOT/"audit_summary.json").read_text(encoding="utf-8"))["planner_record"]["counts"],dtype=np.int64)

def choose(path, counts, seed):
    gi,y,c,seeds=pool_meta(path); rng=np.random.default_rng(seed); out=[]
    for code,n in enumerate(counts,1):
        a=gi[c==code].copy();rng.shuffle(a)
        if len(a)<int(n):raise RuntimeError(f"infeasible {path} code={code}: {len(a)}<{n}")
        out.extend(a[:int(n)].tolist())
    return np.asarray(out,dtype=np.int64)

def write(path, indices, dst):
    indices=np.asarray(indices,dtype=np.int64); order=np.argsort(indices); si=indices[order]; inverse=np.argsort(order)
    with h5py.File(path,"r") as f:
        y=np.asarray(f["known_label"][SEL,:][:,si],dtype=np.uint8)[:,inverse]
        z=np.asarray(f["iq_complex"][:,si])[:,inverse]
        seeds=np.asarray(f["receiver_noise_seed"][:,si]).reshape(1,-1)[:,inverse]
        # Preserve full-power component JNR evidence for audit without copying raw metadata.
        jnr=[]
        d=f["component_nominal_JNR_dB"]
        for i in si:
            jnr.append(float(np.asarray(f[d[0,int(i)]]).reshape(-1).mean()))
    jnr=np.asarray(jnr,dtype=np.float64)[inverse]
    dst.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(dst,"w") as f:
        f.create_dataset("iq_complex",data=z,dtype=z.dtype,compression="gzip",compression_opts=1)
        f.create_dataset("known_label",data=y,dtype="u1")
        f.create_dataset("label_selected",data=y,dtype="u1")
        f.create_dataset("receiver_noise_seed",data=seeds,dtype="f8")
        f.create_dataset("component_nominal_JNR_dB",data=jnr.reshape(1,-1),dtype="f8")
        f.create_dataset("manifest",data=np.asarray([3707764736,2,1,1,3,1],dtype="u4"))
        f.attrs["classes"]="FMB,AMA,SawFM,SinFM,MFT,ISRJ"
        f.attrs["source_contract"]="observable-truth-2026-07-27; full_power_10_30dB_v1"
    return {"path":str(dst),"sha256":sha256(dst),"samples":int(len(indices)),"label_counts":y.sum(1).astype(int).tolist(),"jnr_min":float(np.min(jnr)),"jnr_max":float(np.max(jnr)),"jnr_mean":float(np.mean(jnr))}

def main():
    train_counts=IF3_COUNTS; test_counts=np.full(63,96,dtype=np.int64)
    train_idx=choose(RAW_A_TEST,train_counts,20260905); test_idx=choose(RAW_B_TRAIN,test_counts,20260906)
    # Identity and exact source-index checks are performed before copying.
    _,_,_,sa=pool_meta(RAW_A_TEST); _,_,_,sb=pool_meta(RAW_B_TRAIN)
    if set(sa[train_idx].tolist())&set(sb[test_idx].tolist()):raise AssertionError("source identity overlap")
    OUT.mkdir(parents=True,exist_ok=True); tr=write(RAW_A_TEST,train_idx,OUT/"closedset_6class_train.mat");te=write(RAW_B_TRAIN,test_idx,OUT/"closedset_6class_test.mat")
    with h5py.File(OUT/"closedset_6class_train.mat","r") as f: ytr=np.asarray(f["label_selected"],dtype=np.uint8).T
    with h5py.File(OUT/"closedset_6class_test.mat","r") as f: yte=np.asarray(f["label_selected"],dtype=np.uint8).T
    def d(y):
        c=y@(1<<np.arange(6,dtype=np.uint8));q=np.bincount(c.astype(int),minlength=64)[1:];return {"n":int(len(y)),"label_counts":y.sum(0).astype(int).tolist(),"unique_combinations":int((q>0).sum()),"combo_min":int(q[q>0].min()),"combo_max":int(q[q>0].max()),"effective_if":float(y.sum(0).max()/y.sum(0).min()),"effective_cf":float(q[q>0].max()/q[q>0].min())}
    audit={"schema":"six-class-pure-label-longtail-audit-v3","status":"completed","selected_types":["FMB","AMA","SawFM","SinFM","MFT","ISRJ"],"source_train":str(RAW_A_TEST),"source_test":str(RAW_B_TRAIN),"source_train_sha256":sha256(RAW_A_TEST),"source_test_sha256":sha256(RAW_B_TRAIN),"files":{"closedset_6class_train.mat":tr,"closedset_6class_test.mat":te},"train_distribution":d(ytr),"test_distribution":d(yte),"train_receiver_seed_count":int(len(set(sa[train_idx].tolist()))),"test_receiver_seed_count":int(len(set(sb[test_idx].tolist()))),"train_test_receiver_seed_overlap":int(len(set(sa[train_idx].tolist())&set(sb[test_idx].tolist()))),"train_indices_source":train_idx.tolist(),"test_indices_source":test_idx.tolist(),"replacement":False,"locked_test_used":False,"frozen_e5_modified":False,"evidence_boundary":"fresh-derived-sixclass-data-audit_only"}
    (OUT/"audit_summary.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"status":"completed","output":str(OUT),"train":tr,"test":te},ensure_ascii=False))
if __name__=="__main__":main()
