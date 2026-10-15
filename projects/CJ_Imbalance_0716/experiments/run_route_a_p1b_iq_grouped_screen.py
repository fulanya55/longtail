"""P1b comparison: grouped I/Q convolutions versus capacity-matched real convolutions."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, receiver_seeds, profile_cost, sha256, write_json

class IQStreamConvClassifier(nn.Module):
    """Two-stream I/Q grouped (groups=2) or ordinary real (groups=1) encoder."""
    def __init__(self, groups=2, width=50, num_classes=6):
        super().__init__(); self.groups=int(groups); self.width=int(width)
        c1=2*width; c2=4*width; c3=8*width
        self.features=nn.Sequential(
            nn.Conv1d(2,c1,7,padding=3,groups=groups,bias=False), nn.BatchNorm1d(c1), nn.GELU(),
            nn.MaxPool1d(2), nn.Conv1d(c1,c1,7,padding=3,groups=groups,bias=False), nn.BatchNorm1d(c1), nn.GELU(),
            nn.MaxPool1d(2), nn.Conv1d(c1,c2,5,padding=2,groups=groups,bias=False), nn.BatchNorm1d(c2), nn.GELU(),
            nn.MaxPool1d(2), nn.Conv1d(c2,c3,3,padding=1,groups=groups,bias=False), nn.BatchNorm1d(c3), nn.GELU(),
            nn.Conv1d(c3,c3,1,bias=False), nn.BatchNorm1d(c3), nn.GELU(),
        )
        self.classifier=nn.Sequential(nn.LayerNorm(2*c3),nn.Dropout(0.2),nn.Linear(2*c3,num_classes))
    def forward(self,x):
        z=self.features(x); pooled=torch.cat([z.mean(-1),z.std(-1,unbiased=False)],dim=1); return self.classifier(pooled)

VARIANTS={"F2_full_scratch":("f2",16),"IQ_grouped":("grouped",41),"Real_conv_matched":("real",33)}
SEEDS=(2039,2040,2041)

def set_seed(seed): np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def make_model(name):
    kind,w=VARIANTS[name]
    if kind=="f2":
        from cjlt.models.backbone_adapter import build_backbone
        return build_backbone("dfms_iq",num_classes=6,width=w,dropout=0.2)
    return IQStreamConvClassifier(groups=2 if kind=="grouped" else 1,width=w)
def predict(model,path,indices,device,batch):
    ds=NineClassDataset(path,indices,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True); ys=[];ls=[]
    try:
        model.eval()
        with torch.no_grad():
            for b in loader: ls.append(model(b["signal"].to(device,non_blocking=True)).cpu().numpy()); ys.append(b["label"].numpy())
    finally: ds.close()
    return np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32)
def run_one(args,name,seed,fit,val,power_dev,device,out):
    set_seed(seed); model=make_model(name).to(device); train_path=find_file(args.data_root,30,"train"); ds=NineClassDataset(train_path,fit,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); crit=nn.BCEWithLogitsLoss(); hist=[]; best=-1;best_epoch=0; started=time.time()
    try:
        for epoch in range(1,args.epochs+1):
            model.train(); losses=[]
            for b in loader:
                x=b["signal"].to(device,non_blocking=True); y=b["label"].to(device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=crit(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
            yt,yl=predict(model,train_path,val,device,args.eval_batch_size); vm=metrics(yt,yl); hist.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":vm})
            if vm["macro_aupr"]>best: best=vm["macro_aupr"];best_epoch=epoch;torch.save(model.state_dict(),out/f"{name}__seed_{seed}.pth")
    finally: ds.close()
    model.load_state_dict(torch.load(str(out/f"{name}__seed_{seed}.pth"),map_location=device,weights_only=True)); result={"variant":name,"seed":seed,"history":hist,"best_epoch":best_epoch,"validation":hist[best_epoch-1]["validation"],"power_dev":{},"total_parameters":int(sum(p.numel() for p in model.parameters())),"cost_profile":profile_cost(model,device),"locked_test_used":False,"elapsed_seconds":time.time()-started}
    for power,(path,idx) in power_dev.items(): yt,yl=predict(model,path,idx,device,args.eval_batch_size); result["power_dev"][str(power)]=metrics(yt,yl)
    result["checkpoint_sha256"]=sha256(out/f"{name}__seed_{seed}.pth"); write_json(out/f"{name}__seed_{seed}.json",result); print(json.dumps({"variant":name,"seed":seed,"val_macro_aupr":result["validation"]["macro_aupr"]},ensure_ascii=False),flush=True); return result
def run(args):
    device=torch.device(args.device)
    if device.type!="cuda": raise RuntimeError("P1b requires CUDA")
    out=args.output_root;out.mkdir(parents=True,exist_ok=True); train_path=find_file(args.data_root,30,"train"); pure=pure_six_indices(train_path); perm=np.random.default_rng(args.split_seed).permutation(pure); fit=np.sort(perm[:args.fit_samples]); val=np.sort(perm[args.fit_samples:args.fit_samples+args.val_samples]); used=set(receiver_seeds(train_path,fit)); power_dev={}
    for power in POWERS_DB:
        path=find_file(args.data_root,power,"train"); p=pure_six_indices(path)
        with h5py.File(str(path),"r") as h: rs=np.asarray(h["receiver_noise_seed"][:,:],dtype=np.int64).reshape(-1)
        candidates=p[~np.isin(rs[p],np.fromiter(used,dtype=np.int64))]; idx=np.sort(np.random.default_rng(args.split_seed+power).choice(candidates,size=min(args.power_dev_samples,len(candidates)),replace=False)); power_dev[power]=(path,idx)
    role={"schema":"route-a-p1b-train-only-v1","selected_classes":list(CLASS_NAMES),"fit":{"path":str(train_path),"count":len(fit),"sample_indices":fit.tolist(),"receiver_noise_seed":receiver_seeds(train_path,fit)},"validation":{"path":str(train_path),"count":len(val),"sample_indices":val.tolist(),"receiver_noise_seed":receiver_seeds(train_path,val)},"power_development":{"source_role":"train_only_unseen_identity","count_by_power":{str(p):len(i) for p,(path,i) in power_dev.items()}},"test":{"used":False,"locked_test_used":False}};write_json(out/"role_manifest.json",role)
    rows=[]
    for name in VARIANTS:
        d=out/name;d.mkdir(parents=True,exist_ok=True)
        for seed in args.seeds: rows.append(run_one(args,name,int(seed),fit,val,power_dev,device,d))
    summary={"schema":"route-a-p1b-iq-grouped-summary-v1","status":"completed","locked_test_used":False,"variants":{}}
    for name in VARIANTS:
        c=[r for r in rows if r["variant"]==name];summary["variants"][name]={"parameters":c[0]["total_parameters"],"val_macro_aupr_mean":float(np.mean([r["validation"]["macro_aupr"] for r in c])),"val_macro_aupr_std":float(np.std([r["validation"]["macro_aupr"] for r in c],ddof=1)),"val_combo_mean":float(np.mean([r["validation"]["combination_exact_accuracy"] for r in c])),"power_dev_macro_aupr_mean_by_power":{str(p):float(np.mean([r["power_dev"][str(p)]["macro_aupr"] for r in c])) for p in POWERS_DB}}
    write_json(out/"p1b_iq_grouped_summary.json",summary);write_json(out/"terminal_receipt.json",{"status":"completed","summary":str((out/"p1b_iq_grouped_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False,"script_sha256":sha256(Path(__file__))});print(json.dumps({"status":"completed","summary":str(out/"p1b_iq_grouped_summary.json")},ensure_ascii=False),flush=True)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类"));p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/route_a_p1b_iq_grouped_screen_20260827");p.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS));p.add_argument("--split-seed",type=int,default=260827);p.add_argument("--fit-samples",type=int,default=4096);p.add_argument("--val-samples",type=int,default=2048);p.add_argument("--power-dev-samples",type=int,default=512);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--eval-batch-size",type=int,default=32);p.add_argument("--epochs",type=int,default=2);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--device",default="cuda:0");return p.parse_args()
if __name__=="__main__":run(parse_args())
