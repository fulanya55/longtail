"""Run fixed no-fit H1b all-background weak-VFT screen."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import h5py,numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from cjlt.models.vft_staggered_pri import record_samples,staggered_vft_evidence

def load(path:Path)->np.ndarray:
    with h5py.File(path,"r") as f:r=np.asarray(f["iq_real"],dtype=np.float32);i=np.asarray(f["iq_imag"],dtype=np.float32)
    if r.shape!=(640,record_samples()):raise ValueError(f"unexpected H1b shape {r.shape}")
    return np.stack((r,i),axis=1)
def scores(x:np.ndarray,device:torch.device,batch:int)->tuple[np.ndarray,np.ndarray]:
    a=[];b=[]
    with torch.inference_mode():
      for s in range(0,x.shape[0],batch):
        u,v=staggered_vft_evidence(torch.from_numpy(x[s:s+batch]).to(device));a.append(u.cpu().numpy());b.append(v.cpu().numpy())
    return np.concatenate(a),np.concatenate(b)
def metrics(a,p,ac,pc,draws,seed):
    n=a.size;y=np.r_[np.zeros(n,dtype=np.int8),np.ones(n,dtype=np.int8)];c=np.r_[a,p];k=np.r_[ac,pc];z={"candidate_auroc":float(roc_auc_score(y,c)),"control_auroc":float(roc_auc_score(y,k)),"candidate_aupr":float(average_precision_score(y,c)),"control_aupr":float(average_precision_score(y,k)),"paired_present_gt_absent_fraction":float(np.mean(p>a))};rng=np.random.default_rng(seed);ad=[];pd=[]
    for _ in range(draws):
      q=rng.integers(0,n,n);yy=np.r_[np.zeros(n,dtype=np.int8),np.ones(n,dtype=np.int8)];ad.append(roc_auc_score(yy,np.r_[a[q],p[q]])-roc_auc_score(yy,np.r_[ac[q],pc[q]]));pd.append(average_precision_score(yy,np.r_[a[q],p[q]])-average_precision_score(yy,np.r_[ac[q],pc[q]]))
    z["candidate_minus_control_auroc"]={"mean":z["candidate_auroc"]-z["control_auroc"],"bootstrap_95_ci":[float(np.quantile(ad,.025)),float(np.quantile(ad,.975))]};z["candidate_minus_control_aupr"]={"mean":z["candidate_aupr"]-z["control_aupr"],"bootstrap_95_ci":[float(np.quantile(pd,.025)),float(np.quantile(pd,.975))]};return z
def run(args):
    audit=json.loads(args.audit.read_text(encoding="utf-8"));seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"));
    if not audit.get("identity_audit_passed") or any(seal["test_v3"][k] for k in ("access_authorized","analysis_allowed","used")):raise ValueError("H1b audit or seal invalid")
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"));a,p=load(args.absent_path),load(args.present_path);ca,ka=scores(a,device,args.batch_size);cp,kp=scores(p,device,args.batch_size);m=metrics(ca,cp,ka,kp,args.bootstrap_draws,args.bootstrap_seed);u=m["candidate_minus_control_auroc"];v=m["candidate_minus_control_aupr"]
    checks={"candidate_auroc_ge_0p58":m["candidate_auroc"]>=.58,"candidate_minus_control_auroc_ge_0p04":u["mean"]>=.04,"candidate_minus_control_auroc_ci_lower_gt_zero":u["bootstrap_95_ci"][0]>0,"candidate_minus_control_aupr_ge_0p025":v["mean"]>=.025,"candidate_minus_control_aupr_ci_lower_gt_zero":v["bootstrap_95_ci"][0]>0,"paired_present_gt_absent_fraction_ge_0p56":m["paired_present_gt_absent_fraction"]>=.56,"no_fit":True,"test_v3_still_sealed":True}
    z={"status":"completed","experiment":"RouteA_H1b_VFTStaggeredPRI_AllFiveBackgroundCompositeAttribution","evidence_role":"fresh_paired_project_local_staggered_pri_all_background_composite_physical_identifiability","source_scope":audit["source_scope"],"source_audit":str(args.audit),"source_sha256":{"absent":audit["absent_sha256"],"present":audit["present_sha256"]},"fixed_statistic":{"pulse_count":8,"record_samples":record_samples(),"delay_samples":[100,500],"frequency_hz":[1000,7000,250],"timestamp_permutation":[0,5,2,7,4,1,6,3],"score":"full_grid_maximum_normalized_coherence"},"bootstrap":{"draws":args.bootstrap_draws,"pair_resampling":True,"seed":args.bootstrap_seed},"device":str(device),"metrics":m,"gate":{"passed":bool(all(checks.values())),"checks":checks},"trained_parameters":0,"external_refit":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return z
def parse_args():
    data=Path(r"C:\LJQ_code\干扰数据\h1b_vft_staggered_pri_all_backgrounds_20260803");root=ROOT/"outputs/tapc/route_a_h1b_vft_staggered_pri";p=argparse.ArgumentParser(description=__doc__);p.add_argument("--absent-path",type=Path,default=data/"vft_absent_all_five_backgrounds.mat");p.add_argument("--present-path",type=Path,default=data/"vft_present_weak_8db_all_five_backgrounds.mat");p.add_argument("--audit",type=Path,default=ROOT/"outputs/tapc/route_a_h1b_vft_staggered_pri_generation/paired_identity_audit.json");p.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json");p.add_argument("--output",type=Path,default=root/"route_a_h1b_vft_staggered_pri_summary.json");p.add_argument("--batch-size",type=int,default=32);p.add_argument("--bootstrap-draws",type=int,default=5000);p.add_argument("--bootstrap-seed",type=int,default=20260803);p.add_argument("--device",default=None);return p.parse_args()
if __name__=="__main__":
    x=run(parse_args());print(json.dumps({"status":x["status"],"gate":x["gate"]},ensure_ascii=False,indent=2))
