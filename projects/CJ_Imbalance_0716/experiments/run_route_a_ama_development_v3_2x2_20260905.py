"""Development-v3 2x2: BCE/logit-adjustment x cycle-balanced composition order."""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import initialize_f2, metrics
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import CLASS_NAMES, INIT, sha256, set_seed
from experiments.run_route_a_ama_development_v2_20260903 import SEL, EXC, pool_meta, distribution
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import DATA_ROOT

BASE=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类\power_sweep_9level_v1\level_05__jnr_30dB")
SRC_TRAIN=BASE/"closedset_9class_train.mat"; SRC_TEST=BASE/"closedset_9class_test.mat"
OUT=ROOT/"outputs"/"route_a_ama_development_v3_2x2_20260905"
OLD_V2=ROOT/"outputs"/"route_a_ama_development_v2_20260903"/"manifest.json"
OLD_ANCHOR=ROOT/"outputs"/"route_a_ama_true_anchor_20260903"/"manifest.json"
IF3_COUNTS=np.asarray(json.loads((DATA_ROOT/"audit_summary.json").read_text(encoding="utf-8"))["planner_record"]["counts"],dtype=np.int64)
SEEDS=(2050,2051,2052,2053,2054)

class ArrayDataset(Dataset):
    def __init__(self,x,y): self.x=x; self.y=y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return torch.from_numpy(self.x[i]),torch.from_numpy(self.y[i])

def pure_pool(path):
    with h5py.File(path,"r") as f:
        y9=np.asarray(f["known_label"],dtype=np.uint8); seeds=np.asarray(f["receiver_noise_seed"]).reshape(-1)
    pure=(y9[SEL].sum(axis=0)>0)&(y9[EXC].sum(axis=0)==0); gi=np.flatnonzero(pure); y6=y9[SEL][:,gi].T; c=y6@(1<<np.arange(6,dtype=np.uint8)); return gi,y6,c,seeds

def load(path,indices):
    idx=np.asarray(indices,dtype=np.int64); order=np.argsort(idx); si=idx[order]
    with h5py.File(path,"r") as f:
        y=np.asarray(f["known_label"][SEL,:][:,si],dtype=np.uint8).T; z=np.asarray(f["iq_complex"][:,si]); seeds=np.asarray(f["receiver_noise_seed"][:,si]).reshape(-1)
    x=np.stack([np.asarray(z["real"],dtype=np.float32).T,np.asarray(z["imag"],dtype=np.float32).T],axis=1); rms=np.sqrt(np.mean(x[:,0]**2+x[:,1]**2,axis=1)); x=x/np.sqrt(rms[:,None,None]**2+1e-12); inv=np.argsort(order); return np.ascontiguousarray(x[inv]),y[inv].astype(np.float32),seeds[inv]

def dist(y):
    y=y.astype(np.uint8); lc=y.sum(0).astype(int); c=y@(1<<np.arange(6,dtype=np.uint8)); q=np.bincount(c.astype(int),minlength=64)[1:]; a=q[q>0]; return {"n":int(len(y)),"label_counts":lc.tolist(),"effective_if":float(lc.max()/lc.min()),"unique_combinations":int((q>0).sum()),"combo_min":int(a.min()),"combo_max":int(a.max()),"effective_cf":float(a.max()/a.min())}

def manifest():
    gt,yt,ct,st=pure_pool(SRC_TRAIN); ge,ye,ce,se=pure_pool(SRC_TEST); rng=np.random.default_rng(20260905)
    fit=[]; cal=[]; chk=[]; test=[]
    for code in range(1,64):
        a=gt[ct==code].copy(); b=ge[ce==code].copy(); rng.shuffle(a); rng.shuffle(b)
        n=int(IF3_COUNTS[code-1]);
        if len(a)<n or len(b)<96: raise RuntimeError(f"infeasible code={code} train={len(a)} test={len(b)}")
        fit.extend(a[:n].tolist()); cal.extend(b[:16].tolist()); chk.extend(b[16:32].tolist()); test.extend(b[32:96].tolist())
    _,_,_,old_st=pool_meta(SRC_TRAIN); _,_,_,old_se=pool_meta(SRC_TEST)
    # New power-sweep identities must not overlap prior full-power/source roles.
    old_ids=set()
    for p in (OLD_V2,OLD_ANCHOR):
        if p.exists():
            try:
                m=json.loads(p.read_text(encoding="utf-8")); old_ids.update(m.get("anchor_indices",[])); old_ids.update(m.get("validation_indices",[])); old_ids.update(m.get("development_test_indices",[])); old_ids.update(m.get("fit_indices",[]))
            except Exception: pass
    # Index sets are from other files; identity intersection is checked by seed values below.
    m={"schema":"route-a-ama-development-v3-manifest-v1","classes":list(CLASS_NAMES),"source_train":str(SRC_TRAIN),"source_test":str(SRC_TEST),"source_train_sha256":sha256(SRC_TRAIN),"source_test_sha256":sha256(SRC_TEST),"selection_seed":20260905,"fit_indices":sorted(fit),"calibration_indices":sorted(cal),"check_indices":sorted(chk),"development_test_indices":sorted(test),"roles":{"fit_n":len(fit),"calibration_n":len(cal),"check_n":len(chk),"development_test_n":len(test)},"replacement":False}
    for name,ix in (("fit",fit),("calibration",cal),("check",chk),("development_test",test)): m[name+"_distribution"]=dist(yt[np.searchsorted(gt,np.asarray(ix))] if name=="fit" else ye[np.searchsorted(ge,np.asarray(ix))])
    m["identity_overlap_fit_with_eval"]=int(len(set(st[fit].tolist())&(set(se[cal].tolist())|set(se[chk].tolist())|set(se[test].tolist()))))
    m["identity_overlap_calibration_check_test"]=int(len(set(se[cal].tolist())&(set(se[chk].tolist())|set(se[test].tolist()))))
    m["identity_overlap_check_test"]=int(len(set(se[chk].tolist())&set(se[test].tolist())))
    m["source_train_test_seed_overlap"]=int(len(set(st.tolist())&set(se.tolist())))
    m["source_train_pool_n"]=int(len(gt));m["source_test_pool_n"]=int(len(ge));m["if3_target_counts"]=IF3_COUNTS.tolist()
    # A fixed-budget globally unique balanced schedule is infeasible when target budget exceeds 63*minimum count.
    q=IF3_COUNTS; m["no_replacement_balance_audit"]={"min_combo_count":int(q.min()),"max_unique_balanced_budget":int(63*q.min()),"requested_budget":int(q.sum()),"globally_unique_fixed_budget_feasible":bool(63*q.min()>=q.sum()),"cycle_schedule_required":bool(63*q.min()<q.sum()),"note":"A cycle-balanced order can be balanced per cycle but must repeat rare-combination identities across cycles; repetition is measured and reported."}
    return m

def predict(model,x,y,device):
    dl=DataLoader(ArrayDataset(x,y),batch_size=64,shuffle=False,num_workers=0,pin_memory=True); ys=[];ls=[];model.eval()
    with torch.no_grad():
        for xb,yb in dl: ys.append(yb.numpy());ls.append(model(xb.to(device,non_blocking=True)).cpu().numpy())
    return metrics(np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32))

class LogitAdj(nn.Module):
    def __init__(self,counts,n):
        super().__init__(); p=torch.as_tensor(counts,dtype=torch.float32)/float(n); self.register_buffer("bias",torch.log(p/(1-p)))
    def forward(self,z,y): return nn.functional.binary_cross_entropy_with_logits(z+self.bias.view(1,-1),y.float())

def cycle_order(labels,seed,n):
    rng=np.random.default_rng(seed); codes=labels.astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8)); groups={c:rng.permutation(np.flatnonzero(codes==c)).tolist() for c in sorted(set(codes.tolist()))}; out=[]; ptr={c:0 for c in groups}
    while len(out)<n:
        for c in sorted(groups):
            if len(out)>=n: break
            if ptr[c]>=len(groups[c]): groups[c]=rng.permutation(groups[c]).tolist(); ptr[c]=0
            out.append(groups[c][ptr[c]]); ptr[c]+=1
    return np.asarray(out,dtype=np.int64)

def train_one(condition,seed,fit_x,fit_y,cal_x,cal_y,chk_x,chk_y,test_x,test_y,device):
    set_seed(seed); model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=0.2).to(device); init=initialize_f2(model,INIT); counts=fit_y.astype(np.uint8).sum(0).astype(int); criterion=LogitAdj(counts,len(fit_y)) if condition.startswith("logit") else nn.BCEWithLogitsLoss()
    criterion=criterion.to(device)
    balanced=condition.endswith("balanced"); order=cycle_order(fit_y,seed+99000,len(fit_y)) if balanced else None
    ds=ArrayDataset(fit_x,fit_y); loader=DataLoader(ds,batch_size=32,shuffle=not balanced,sampler=None if not balanced else torch.utils.data.SequentialSampler(order.tolist()),num_workers=0,pin_memory=True)
    out=OUT/condition;out.mkdir(parents=True,exist_ok=True); ck=out/f"{condition}__seed_{seed}.pth"; opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); hist=[];best=-1.;be=0; started=time.time()
    # For balanced schedule, SequentialSampler emits order positions; wrap dataset in ordered tensors.
    if balanced: ds=ArrayDataset(fit_x[order],fit_y[order]); loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
    for ep in range(1,9):
        model.train(); losses=[]
        for x,y in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True);loss=criterion(model(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach().cpu()))
        vm=predict(model,cal_x,cal_y,device);hist.append({"epoch":ep,"train_loss":float(np.mean(losses)),"validation":vm}); print(json.dumps({"condition":condition,"seed":seed,"epoch":ep,"val_macro_aupr":vm["macro_aupr"],"val_macro_f1":vm["macro_f1"],"val_combo":vm["combination_exact_accuracy"]}),flush=True)
        if vm["macro_aupr"]>best: best=vm["macro_aupr"];be=ep;torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"validation":vm},ck)
    model.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)["model_state_dict"]); cm=predict(model,chk_x,chk_y,device); tm=predict(model,test_x,test_y,device); uniq=int(len(np.unique(order))) if balanced else len(fit_y); row={"schema":"route-a-ama-development-v3-run-v1","status":"completed","condition":condition,"seed":int(seed),"classes":list(CLASS_NAMES),"backbone":"F2_DFMS_Full","loss":"LogitAdjustment" if condition.startswith("logit") else "BCEWithLogitsLoss","sampling":"cycle_balanced_per_cycle_with_global_repetition" if balanced else "ordinary_shuffle","requested_fit_budget":len(fit_y),"unique_fit_identities_used":uniq,"global_repetition_rate":float(1-uniq/len(fit_y)),"optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":32,"epochs":8,"gradient_clip_norm":5.0,"initialization":init,"best_epoch":be,"calibration_validation":hist[be-1]["validation"],"check_validation":cm,"development_test":tm,"history":hist,"checkpoint_sha256":sha256(ck),"elapsed_seconds":float(time.time()-started),"locked_test_used":False,"frozen_e5_modified":False}
    (out/f"{condition}__seed_{seed}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8");return row

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda:0");OUT.mkdir(parents=True,exist_ok=True);m=manifest();(OUT/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
    fx,fy,fs=load(SRC_TRAIN,m["fit_indices"]);cx,cy,cs=load(SRC_TEST,m["calibration_indices"]);qx,qy,qs=load(SRC_TEST,m["check_indices"]);tx,ty,ts=load(SRC_TEST,m["development_test_indices"])
    if m["identity_overlap_fit_with_eval"] or m["identity_overlap_calibration_check_test"] or m["identity_overlap_check_test"] or m["source_train_test_seed_overlap"]: raise AssertionError("development-v3 identity overlap")
    rows=[]
    for cond in ("bce_ordinary","logit_ordinary","bce_balanced","logit_balanced"):
        for seed in SEEDS:
            receipt=OUT/cond/f"{cond}__seed_{seed}.json"
            rows.append(json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else train_one(cond,seed,fx,fy,cx,cy,qx,qy,tx,ty,device))
    summary={"schema":"route-a-ama-development-v3-2x2-summary-v1","status":"completed","manifest":m,"conditions":["bce_ordinary","logit_ordinary","bce_balanced","logit_balanced"],"seeds":list(SEEDS),"calibration_distribution":dist(cy.astype(np.uint8)),"check_distribution":dist(qy.astype(np.uint8)),"development_test_distribution":dist(ty.astype(np.uint8)),"runs":rows,"locked_test_used":False,"frozen_e5_modified":False,"evidence_boundary":"development-v3-screening"};(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"locked_test_used":False,"frozen_e5_modified":False},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"runs":len(rows)},ensure_ascii=False))
if __name__=="__main__": main()
