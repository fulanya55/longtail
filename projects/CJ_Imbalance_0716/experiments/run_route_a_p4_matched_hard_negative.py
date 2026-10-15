"""P4 minimal screen: matched hard-negative ranking for six-class Route-A."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from cjlt.data.longtail_manifest import build_mixed_longtail, compute_weak_component_mask, sample_id_sha256
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, profile_cost, receiver_seeds, sha256, write_json

VARIANTS=("F2_BCE","F2_BCE_matchedHN"); SEEDS=(2039,2040,2041)
def set_seed(seed): np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class PairDataset(Dataset):
    def __init__(self,path,indices,pairs):
        self.base=NineClassDataset(path,indices,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); self.pairs=np.asarray(pairs,dtype=np.int64)
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        a=self.base[int(i)]; b=self.base[int(self.pairs[int(i)])]
        return {"signal":a["signal"],"label":a["label"],"pair_signal":b["signal"],"pair_label":b["label"]}
    def close(self): self.base.close()

def predict(model,path,idx,device,batch):
    ds=NineClassDataset(path,idx,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True); ys=[];ls=[]
    try:
        model.eval()
        with torch.no_grad():
            for b in dl: ls.append(model(b["signal"].to(device,non_blocking=True)).cpu().numpy()); ys.append(b["label"].numpy())
    finally: ds.close()
    return np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32)

def load_shared(model,ckpt):
    p=torch.load(str(ckpt),map_location="cpu",weights_only=False); s=p.get("model_state_dict",p); shared={}
    for k,v in s.items():
        if k.startswith("backbone."): k=k[9:]
        if not k.startswith("classifier."): shared[k]=v
    missing,unexpected=model.load_state_dict(shared,strict=False); missing=[k for k in missing if not k.startswith("classifier.")]; unexpected=[k for k in unexpected if not k.startswith(("label_feature_bank.","prototype_memory."))]
    if missing or unexpected: raise RuntimeError(f"F2 initialization mismatch: {missing[:3]} {unexpected[:3]}")
    return {"checkpoint":str(ckpt.resolve()),"checkpoint_sha256":sha256(ckpt),"best_epoch":p.get("best_epoch")}

def make_pairs(labels,power,seed,conditions=None):
    n=len(labels); pairs=np.arange(n,dtype=np.int64); rng=np.random.default_rng(seed); card=labels.sum(1); total=np.where(labels,power,0).sum(1)
    if conditions is not None:
        conditions=np.asarray(conditions,dtype=float); med=np.zeros(conditions.shape[1],dtype=float); scale=np.ones(conditions.shape[1],dtype=float)
        for c in range(conditions.shape[1]):
            finite=np.isfinite(conditions[:,c])
            if finite.any():
                med[c]=float(np.median(conditions[finite,c])); mad=float(np.median(np.abs(conditions[finite,c]-med[c]))*1.4826); scale[c]=mad if np.isfinite(mad) and mad>1e-9 else 1.0
        cond=np.where(np.isfinite(conditions),(conditions-med)/scale,0.0)
    else: cond=None
    for i in range(n):
        cand=np.flatnonzero((card==card[i]) & (np.arange(n)!=i) & np.any(labels!=labels[i],axis=1))
        if len(cand)==0: cand=np.flatnonzero((np.arange(n)!=i) & np.any(labels!=labels[i],axis=1))
        if len(cand):
            dist=np.abs(total[cand]-total[i]) if cond is None else np.linalg.norm(cond[cand]-cond[i],axis=1); best=cand[np.flatnonzero(dist==dist.min())]; pairs[i]=int(rng.choice(best))
    return pairs

def run_one(args,name,seed,path,indices,pairs,val,power_dev,out,meta):
    set_seed(seed); model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).to(args.device); init=load_shared(model,args.f2_checkpoint)
    ds=PairDataset(path,indices,pairs); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); hist=[]; best=-1.; best_ep=0; started=time.time()
    try:
        for ep in range(1,args.epochs+1):
            model.train(); losses=[]; hn_losses=[]
            for b in dl:
                x=b["signal"].to(args.device,non_blocking=True); y=b["label"].to(args.device,non_blocking=True); xp=b["pair_signal"].to(args.device,non_blocking=True); yp=b["pair_label"].to(args.device,non_blocking=True); opt.zero_grad(set_to_none=True); la=model(x); lb=model(xp); lc=nn.functional.binary_cross_entropy_with_logits(la,y)
                diff=y-yp; mask=diff.abs()>0
                if name=="F2_BCE_matchedHN" and bool(mask.any()):
                    # For each symmetric-difference label, the active sample must score higher.
                    sign=torch.where(diff>0,torch.ones_like(diff),-torch.ones_like(diff)); lh=torch.relu(args.margin-sign*(la-lb))[mask].mean(); loss=lc+args.hn_weight*lh
                else: lh=la.sum()*0; loss=lc
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu())); hn_losses.append(float(lh.detach().cpu()))
            yt,yl=predict(model,path,val,args.device,args.eval_batch_size); vm=metrics(yt,yl); hist.append({"epoch":ep,"train_loss":float(np.mean(losses)),"matched_hn_loss":float(np.mean(hn_losses)),"validation":vm})
            if vm["macro_aupr"]>best: best=vm["macro_aupr"]; best_ep=ep; torch.save(model.state_dict(),out/f"{name}__seed_{seed}.pth")
    finally: ds.close()
    model.load_state_dict(torch.load(str(out/f"{name}__seed_{seed}.pth"),map_location=args.device,weights_only=True)); r={"variant":name,"seed":seed,"manifest":meta,"initialization":init,"history":hist,"best_epoch":best_ep,"validation":hist[best_ep-1]["validation"],"power_dev":{},"total_parameters":int(sum(p.numel() for p in model.parameters())),"cost_profile":profile_cost(model,args.device),"hn_weight":args.hn_weight,"margin":args.margin,"locked_test_used":False,"elapsed_seconds":time.time()-started}
    for p,(pp,ii) in power_dev.items(): yt,yl=predict(model,pp,ii,args.device,args.eval_batch_size); r["power_dev"][str(p)]=metrics(yt,yl)
    r["checkpoint_sha256"]=sha256(out/f"{name}__seed_{seed}.pth"); write_json(out/f"{name}__seed_{seed}.json",r); print(json.dumps({"variant":name,"seed":seed,"val_macro_aupr":r["validation"]["macro_aupr"],"val_combo":r["validation"]["combination_exact_accuracy"]},ensure_ascii=False),flush=True); return r

def run(args):
    dev=torch.device(args.device)
    if dev.type!="cuda": raise RuntimeError("P4 requires CUDA")
    args.device=dev; args.output_root.mkdir(parents=True,exist_ok=True); path=find_file(args.data_root,30,"train"); pure=pure_six_indices(path); perm=np.random.default_rng(args.split_seed).permutation(pure); fit=np.sort(perm[:args.fit_samples]); val=np.sort(perm[args.fit_samples:args.fit_samples+args.val_samples])
    # Decode only the fit pool's six-class labels and effective component powers.
    import h5py
    with h5py.File(str(path),'r') as h:
        labels9=np.asarray(h['known_label'][:,fit],dtype=np.uint8).T; refs=h['component_effective_JNR_dB'][0,fit]; labels6=labels9[:,list(SELECTED_INDICES)]; power=np.zeros_like(labels6,dtype=float)
        conditions=np.stack([np.asarray(h[k][0,fit],dtype=float).reshape(-1) for k in ('target_range_m','target_delay_s','target_measured_power_dB','target_present')],axis=1)
        for r,ref in enumerate(refs):
            v=np.asarray(h[ref][:],dtype=float).reshape(-1); active=np.flatnonzero(labels9[r]);
            if len(v)!=len(active): raise RuntimeError('power alignment mismatch')
            for pos,g in enumerate(active.tolist()):
                if g in SELECTED_INDICES: power[r,list(SELECTED_INDICES).index(g)]=v[pos]
    used=set(receiver_seeds(path,np.sort(np.concatenate([fit,val])))); power_dev={}
    for p in POWERS_DB:
        pp=find_file(args.data_root,p,'train'); pi=pure_six_indices(pp); rs=np.asarray(receiver_seeds(pp,pi),dtype=np.int64); cand=pi[~np.isin(rs,np.fromiter(used,dtype=np.int64))]; ii=np.sort(np.random.default_rng(args.split_seed+p).choice(cand,size=min(args.power_dev_samples,len(cand)),replace=False)); power_dev[p]=(pp,ii)
    role={"schema":"route-a-p4-matched-hard-negative-role-manifest-v1","selected_classes":list(CLASS_NAMES),"excluded_classes":["SN","VFT","RangeFakeTarget"],"fit":{"path":str(path),"count":int(len(fit)),"sample_indices_sha256":sample_id_sha256(fit),"receiver_noise_seed":sorted(set(receiver_seeds(path,fit)))},"validation":{"path":str(path),"count":int(len(val)),"sample_indices_sha256":sample_id_sha256(val),"receiver_noise_seed":sorted(set(receiver_seeds(path,val)))},"power_development":{"source_role":"train_only_unseen_identity","count_by_power":{str(p):len(i) for p,(pp,i) in power_dev.items()}},"test":{"used":False,"locked_test_used":False}}; write_json(args.output_root/'role_manifest.json',role)
    rows=[]
    for seed in args.seeds:
        manifest=build_mixed_longtail(labels6,np.arange(len(fit)),power,args.label_if,args.combo_if,args.weak_threshold_db,int(seed),sample_id_sha256(fit)); local=manifest.selected_indices.astype(np.int64); orig=fit[local]; order=np.argsort(orig); orig=orig[order]; sel_labels=labels6[local][order]; sel_power=power[local][order]; sel_conditions=conditions[local][order]; pairs=make_pairs(sel_labels,sel_power,seed,sel_conditions); meta=manifest.to_dict(); meta.update({"selected_original_indices_sha256":sample_id_sha256(orig),"selected_count":int(len(orig)),"pair_count":int(np.sum(pairs!=np.arange(len(orig)))),"pair_rule":"same active-label cardinality, different combination, nearest robust-standardized target range/delay/power/presence; receiver_noise_seed unique per sample"}); write_json(args.output_root/f"longtail_hn_manifest_seed_{seed}.json",meta)
        for name in VARIANTS:
            out=args.output_root/name; out.mkdir(parents=True,exist_ok=True); rows.append(run_one(args,name,int(seed),path,orig,pairs,val,power_dev,out,meta))
    summary={"schema":"route-a-p4-matched-hard-negative-summary-v1","status":"completed","locked_test_used":False,"protocol":{"epochs":args.epochs,"seeds":list(args.seeds),"hn_weight":args.hn_weight,"margin":args.margin,"label_if":args.label_if,"combo_if":args.combo_if},"variants":{}}
    for name in VARIANTS:
        rr=[r for r in rows if r['variant']==name]; summary['variants'][name]={"parameters":rr[0]['total_parameters'],"val_macro_aupr_mean":float(np.mean([r['validation']['macro_aupr'] for r in rr])),"val_macro_aupr_std":float(np.std([r['validation']['macro_aupr'] for r in rr],ddof=1)),"val_macro_f1_mean":float(np.mean([r['validation']['macro_f1'] for r in rr])),"val_combo_mean":float(np.mean([r['validation']['combination_exact_accuracy'] for r in rr])),"power_dev_macro_aupr_mean_by_power":{str(p):float(np.mean([r['power_dev'][str(p)]['macro_aupr'] for r in rr])) for p in POWERS_DB},"power_dev_combo_mean_by_power":{str(p):float(np.mean([r['power_dev'][str(p)]['combination_exact_accuracy'] for r in rr])) for p in POWERS_DB}}
    write_json(args.output_root/'p4_matched_hn_summary.json',summary); write_json(args.output_root/'terminal_receipt.json',{"status":"completed","summary":str((args.output_root/'p4_matched_hn_summary.json').resolve()),"locked_test_used":False,"test_files_opened":False,"script_sha256":sha256(Path(__file__))}); print(json.dumps({"status":"completed","summary":str(args.output_root/'p4_matched_hn_summary.json')},ensure_ascii=False),flush=True)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path(r'E:\论文相关\干扰数据\复合干扰数据0716\九类')); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p4_matched_hn_20260828'); p.add_argument('--f2-checkpoint',type=Path,default=PROJECT_ROOT/'outputs/route_a_c1_baseline_full_20260826/best_model.pth'); p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS)); p.add_argument('--split-seed',type=int,default=260829); p.add_argument('--fit-samples',type=int,default=4096); p.add_argument('--val-samples',type=int,default=2048); p.add_argument('--power-dev-samples',type=int,default=256); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--eval-batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=2); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--label-if',type=float,default=5.0); p.add_argument('--combo-if',type=float,default=5.0); p.add_argument('--weak-threshold-db',type=float,default=10.0); p.add_argument('--hn-weight',type=float,default=.1); p.add_argument('--margin',type=float,default=.25); p.add_argument('--device',default='cuda:0'); return p.parse_args()
if __name__=='__main__': run(parse_args())
