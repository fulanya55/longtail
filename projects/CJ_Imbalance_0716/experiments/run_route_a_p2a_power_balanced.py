"""P2a: compare 30 dB-only fitting with equal-power fitting on train-only data."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, receiver_seeds, profile_cost, sha256, write_json

class MultiPowerDataset(Dataset):
    def __init__(self, parts):
        self.parts=[]; self.offsets=[]; total=0
        for path,idx in parts:
            ds=NineClassDataset(path,np.asarray(idx,dtype=np.int64),label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); self.parts.append(ds); total+=len(ds); self.offsets.append(total)
        self.total=total
    def __len__(self): return self.total
    def __getitem__(self,i):
        part=next(j for j,o in enumerate(self.offsets) if i<o); prev=0 if part==0 else self.offsets[part-1]; return self.parts[part][i-prev]
    def close(self):
        for ds in self.parts: ds.close()

VARIANTS=("F2_30db_only","F2_curriculum","F2_power_balanced")
SEEDS=(2039,2040,2041)
def set_seed(s): np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def predict(model,path,idx,device,batch):
    ds=NineClassDataset(path,idx,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True); ys=[];ls=[]
    try:
        model.eval()
        with torch.no_grad():
            for b in dl: ls.append(model(b["signal"].to(device,non_blocking=True)).cpu().numpy());ys.append(b["label"].numpy())
    finally: ds.close()
    return np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32)
def run_one(args,name,seed,fit_parts,val_path,val_idx,power_dev,device,out):
    set_seed(seed); model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=0.2).to(device); ds=MultiPowerDataset(fit_parts[name]);dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay);crit=nn.BCEWithLogitsLoss();hist=[];best=-1;best_epoch=0;started=time.time()
    try:
        for ep in range(1,args.epochs+1):
            model.train();losses=[]
            for b in dl:
                x=b["signal"].to(device,non_blocking=True);y=b["label"].to(device,non_blocking=True);opt.zero_grad(set_to_none=True);loss=crit(model(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach().cpu()))
            yt,yl=predict(model,val_path,val_idx,device,args.eval_batch_size);vm=metrics(yt,yl);hist.append({"epoch":ep,"train_loss":float(np.mean(losses)),"validation":vm})
            if vm["macro_aupr"]>best:best=vm["macro_aupr"];best_epoch=ep;torch.save(model.state_dict(),out/f"{name}__seed_{seed}.pth")
    finally: ds.close()
    model.load_state_dict(torch.load(str(out/f"{name}__seed_{seed}.pth"),map_location=device,weights_only=True));r={"variant":name,"seed":seed,"history":hist,"best_epoch":best_epoch,"validation":hist[best_epoch-1]["validation"],"power_dev":{},"total_parameters":sum(p.numel() for p in model.parameters()),"cost_profile":profile_cost(model,device),"locked_test_used":False,"elapsed_seconds":time.time()-started}
    for p,(path,idx) in power_dev.items():yt,yl=predict(model,path,idx,device,args.eval_batch_size);r["power_dev"][str(p)]=metrics(yt,yl)
    r["checkpoint_sha256"]=sha256(out/f"{name}__seed_{seed}.pth");write_json(out/f"{name}__seed_{seed}.json",r);print(json.dumps({"variant":name,"seed":seed,"val_macro_aupr":r["validation"]["macro_aupr"]},ensure_ascii=False),flush=True);return r
def run(args):
    device=torch.device(args.device)
    if device.type!="cuda":raise RuntimeError("P2a requires CUDA")
    out=args.output_root;out.mkdir(parents=True,exist_ok=True);paths={p:find_file(args.data_root,p,"train") for p in POWERS_DB}; base=paths[30]; pure=pure_six_indices(base);rng=np.random.default_rng(args.split_seed);perm=rng.permutation(pure);fit30=np.sort(perm[:args.fit_samples]);val=np.sort(perm[args.fit_samples:args.fit_samples+args.val_samples]);fit_seeds=set(receiver_seeds(base,fit30));
    # Use one fixed identity set at every power so the balanced budget is 9*fit_per_power.
    seed_pool=np.asarray(sorted(fit_seeds)[:args.fit_per_power],dtype=np.int64); fit_parts={"F2_30db_only":[(base,fit30[:args.fit_samples])],"F2_curriculum":[],"F2_power_balanced":[]}
    for p,path in paths.items():
        with h5py.File(str(path),"r") as h: rs=np.asarray(h["receiver_noise_seed"][:,:],dtype=np.int64).reshape(-1)
        index_map={int(s):int(i) for i,s in enumerate(rs)}
        balanced_idx=np.asarray([index_map[int(s)] for s in seed_pool],dtype=np.int64)
        fit_parts["F2_power_balanced"].append((path,balanced_idx))
        if p in (25,35):
            fit_parts["F2_curriculum"].append((path,balanced_idx[:args.curriculum_per_neighbor]))
    fit_parts["F2_curriculum"].insert(0,(base,fit30[:args.curriculum_main]))
    # Keep all power-development identities disjoint from the 30 dB fit role.
    used=set(int(s) for s in fit_seeds);power_dev={}
    for p,path in paths.items():
        purep=pure_six_indices(path)
        with h5py.File(str(path),"r") as h:rs=np.asarray(h["receiver_noise_seed"][:,:],dtype=np.int64).reshape(-1)
        cand=purep[~np.isin(rs[purep],np.fromiter(used,dtype=np.int64))];idx=np.sort(np.random.default_rng(args.split_seed+p).choice(cand,size=min(args.power_dev_samples,len(cand)),replace=False));power_dev[p]=(path,idx)
    role={"schema":"route-a-p2a-power-balanced-v1","selected_classes":list(CLASS_NAMES),"fit":{"30db_only_count":len(fit_parts["F2_30db_only"][0][1]),"power_balanced_count":sum(len(i) for _,i in fit_parts["F2_power_balanced"]),"fit_per_power":args.fit_per_power},"validation":{"path":str(base),"count":len(val),"sample_indices":val.tolist(),"receiver_noise_seed":receiver_seeds(base,val)},"power_development":{"source_role":"train_only_unseen_identity","count_by_power":{str(p):len(i) for p,(path,i) in power_dev.items()}},"test":{"used":False,"locked_test_used":False}};write_json(out/"role_manifest.json",role)
    rows=[]
    for name in VARIANTS:
        d=out/name;d.mkdir(parents=True,exist_ok=True)
        for seed in args.seeds:rows.append(run_one(args,name,int(seed),fit_parts,base,val,power_dev,device,d))
    summary={"schema":"route-a-p2a-power-balanced-summary-v1","status":"completed","locked_test_used":False,"variants":{}}
    for name in VARIANTS:
        c=[r for r in rows if r["variant"]==name];summary["variants"][name]={"parameters":c[0]["total_parameters"],"val_macro_aupr_mean":float(np.mean([r["validation"]["macro_aupr"] for r in c])),"val_macro_aupr_std":float(np.std([r["validation"]["macro_aupr"] for r in c],ddof=1)),"val_combo_mean":float(np.mean([r["validation"]["combination_exact_accuracy"] for r in c])),"power_dev_macro_aupr_mean_by_power":{str(p):float(np.mean([r["power_dev"][str(p)]["macro_aupr"] for r in c])) for p in POWERS_DB}}
    write_json(out/"p2a_power_balanced_summary.json",summary);write_json(out/"terminal_receipt.json",{"status":"completed","summary":str((out/"p2a_power_balanced_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False,"script_sha256":sha256(Path(__file__))});print(json.dumps({"status":"completed","summary":str(out/"p2a_power_balanced_summary.json")},ensure_ascii=False),flush=True)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"));p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/route_a_p2a_power_balanced_20260827");p.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS));p.add_argument("--split-seed",type=int,default=260827);p.add_argument("--fit-samples",type=int,default=4096);p.add_argument("--fit-per-power",type=int,default=456);p.add_argument("--curriculum-main",type=int,default=3072);p.add_argument("--curriculum-per-neighbor",type=int,default=512);p.add_argument("--val-samples",type=int,default=2048);p.add_argument("--power-dev-samples",type=int,default=512);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--eval-batch-size",type=int,default=32);p.add_argument("--epochs",type=int,default=2);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--device",default="cuda:0");return p.parse_args()
if __name__=="__main__":run(parse_args())
