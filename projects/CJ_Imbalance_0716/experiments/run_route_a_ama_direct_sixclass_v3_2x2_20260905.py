"""2x2 five-seed confirmation on the freshly derived direct six-class IF3 dataset."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import initialize_f2,metrics
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import INIT,CLASS_NAMES,sha256,set_seed
DATA=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA")
TRAIN=DATA/"closedset_6class_train.mat"; TEST=DATA/"closedset_6class_test.mat"; OUT=ROOT/"outputs"/"route_a_ama_direct_sixclass_v3_2x2_20260905"
SEEDS=(2050,2051,2052,2053,2054)
class DS(Dataset):
 def __init__(self,x,y):self.x=x;self.y=y
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return torch.from_numpy(self.x[i]),torch.from_numpy(self.y[i])
def load(path,indices=None):
 with h5py.File(path,"r") as f:
  n=f["label_selected"].shape[1]; idx=np.arange(n,dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64); order=np.argsort(idx);si=idx[order]; y=np.asarray(f["label_selected"][:,si],dtype=np.uint8).T; z=np.asarray(f["iq_complex"][:,si]); seeds=np.asarray(f["receiver_noise_seed"][:,si]).reshape(-1)
 x=np.stack([np.asarray(z["real"],dtype=np.float32).T,np.asarray(z["imag"],dtype=np.float32).T],axis=1); rms=np.sqrt(np.mean(x[:,0]**2+x[:,1]**2,axis=1));x=x/np.sqrt(rms[:,None,None]**2+1e-12); inv=np.argsort(order);return np.ascontiguousarray(x[inv]),y[inv].astype(np.float32),seeds[inv]
def dist(y):
 y=y.astype(np.uint8);c=y@(1<<np.arange(6,dtype=np.uint8));q=np.bincount(c.astype(int),minlength=64)[1:];a=q[q>0];return {"n":int(len(y)),"label_counts":y.sum(0).astype(int).tolist(),"unique_combinations":int((q>0).sum()),"combo_min":int(a.min()),"combo_max":int(a.max()),"effective_if":float(y.sum(0).max()/y.sum(0).min()),"effective_cf":float(a.max()/a.min())}
def manifest():
 _,yt,st=load(TRAIN);_,ye,se=load(TEST);c=ye.astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8));rng=np.random.default_rng(20260905);cal=[];chk=[];test=[]
 for code in range(1,64):
  a=np.flatnonzero(c==code);rng.shuffle(a);cal+=a[:16].tolist();chk+=a[16:32].tolist();test+=a[32:96].tolist()
 m={"schema":"route-a-ama-direct-sixclass-v3-manifest-v1","data_root":str(DATA),"train_sha256":sha256(TRAIN),"test_sha256":sha256(TEST),"classes":list(CLASS_NAMES),"selection_seed":20260905,"fit_indices":list(range(len(yt))),"calibration_indices":sorted(cal),"check_indices":sorted(chk),"development_test_indices":sorted(test),"roles":{"fit_n":len(yt),"calibration_n":len(cal),"check_n":len(chk),"development_test_n":len(test)},"fit_distribution":dist(yt),"calibration_distribution":dist(ye[np.asarray(cal)]),"check_distribution":dist(ye[np.asarray(chk)]),"development_test_distribution":dist(ye[np.asarray(test)]),"replacement":False,"train_test_seed_overlap":int(len(set(st.tolist())&set(se.tolist()))),"locked_test_used":False,"frozen_e5_modified":False};return m
class LogitAdj(nn.Module):
 def __init__(self,counts,n):
  super().__init__();p=torch.as_tensor(counts,dtype=torch.float32)/float(n);self.register_buffer("bias",torch.log(p/(1-p)))
 def forward(self,z,y):return nn.functional.binary_cross_entropy_with_logits(z+self.bias.view(1,-1),y.float())
def cycle_order(y,seed,n):
 rng=np.random.default_rng(seed);c=y.astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8));g={k:rng.permutation(np.flatnonzero(c==k)).tolist() for k in sorted(set(c.tolist()))};ptr={k:0 for k in g};o=[]
 while len(o)<n:
  for k in sorted(g):
   if len(o)>=n:break
   if ptr[k]>=len(g[k]):g[k]=rng.permutation(g[k]).tolist();ptr[k]=0
   o.append(g[k][ptr[k]]);ptr[k]+=1
 return np.asarray(o,dtype=np.int64)
def predict(model,x,y,device):
 out=[];model.eval()
 with torch.no_grad():
  for xb,yb in DataLoader(DS(x,y),batch_size=64,shuffle=False,num_workers=0,pin_memory=True):out.append(model(xb.to(device,non_blocking=True)).cpu().numpy())
 return metrics(y.astype(np.uint8),np.concatenate(out).astype(np.float32))
def train_one(cond,seed,fx,fy,cx,cy,qx,qy,tx,ty,device):
 set_seed(seed);model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).to(device);init=initialize_f2(model,INIT);counts=fy.astype(np.uint8).sum(0).astype(int);criterion=LogitAdj(counts,len(fy)) if cond.startswith("logit") else nn.BCEWithLogitsLoss();criterion=criterion.to(device);balanced=cond.endswith("balanced");order=cycle_order(fy,seed+99000,len(fy)) if balanced else np.arange(len(fy)); ds=DS(fx[order],fy[order]);loader=DataLoader(ds,batch_size=32,shuffle=not balanced,num_workers=0,pin_memory=True);od=OUT/cond;od.mkdir(parents=True,exist_ok=True);ck=od/f"{cond}__seed_{seed}.pth";opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);hist=[];best=-1.;be=0;start=time.time()
 for ep in range(1,9):
  model.train();ls=[]
  for x,y in loader:
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True);z=model(x);loss=criterion(z,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();ls.append(float(loss.detach().cpu()))
  vm=predict(model,cx,cy,device);hist.append({"epoch":ep,"train_loss":float(np.mean(ls)),"validation":vm});print(json.dumps({"condition":cond,"seed":seed,"epoch":ep,"val_macro_aupr":vm["macro_aupr"],"val_macro_f1":vm["macro_f1"],"val_combo":vm["combination_exact_accuracy"]}),flush=True)
  if vm["macro_aupr"]>best:best=vm["macro_aupr"];be=ep;torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"validation":vm},ck)
 model.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)["model_state_dict"]);qm=predict(model,qx,qy,device);tm=predict(model,tx,ty,device);uniq=int(len(np.unique(order)));row={"schema":"route-a-ama-direct-sixclass-v3-run-v1","status":"completed","condition":cond,"seed":int(seed),"classes":list(CLASS_NAMES),"backbone":"F2_DFMS_Full","loss":"LogitAdjustment" if cond.startswith("logit") else "BCEWithLogitsLoss","sampling":"cycle_balanced_per_cycle_with_global_repetition" if balanced else "ordinary_shuffle","fit_budget":len(fy),"unique_fit_identities_used":uniq,"global_repetition_rate":float(1-uniq/len(fy)),"optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":32,"epochs":8,"gradient_clip_norm":5.0,"initialization":init,"best_epoch":be,"calibration_validation":hist[be-1]["validation"],"check_validation":qm,"development_test":tm,"history":hist,"checkpoint_sha256":sha256(ck),"elapsed_seconds":float(time.time()-start),"locked_test_used":False,"frozen_e5_modified":False};(od/f"{cond}__seed_{seed}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8");return row
def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 device=torch.device("cuda:0");OUT.mkdir(parents=True,exist_ok=True);m=manifest();(OUT/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8");fx,fy,_=load(TRAIN,m["fit_indices"]);cx,cy,_=load(TEST,m["calibration_indices"]);qx,qy,_=load(TEST,m["check_indices"]);tx,ty,_=load(TEST,m["development_test_indices"]);rows=[]
 if m["train_test_seed_overlap"]:raise AssertionError("train/test identity overlap")
 for cond in ("bce_ordinary","logit_ordinary","bce_balanced","logit_balanced"):
  for seed in SEEDS:
   r=OUT/cond/f"{cond}__seed_{seed}.json";rows.append(json.loads(r.read_text(encoding="utf-8")) if r.exists() else train_one(cond,seed,fx,fy,cx,cy,qx,qy,tx,ty,device))
 summary={"schema":"route-a-ama-direct-sixclass-v3-summary-v1","status":"completed","manifest":m,"conditions":["bce_ordinary","logit_ordinary","bce_balanced","logit_balanced"],"seeds":list(SEEDS),"runs":rows,"locked_test_used":False,"frozen_e5_modified":False,"evidence_boundary":"fresh-sixclass-development-v3; test_role_frozen_diagnostic"};(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"locked_test_used":False,"frozen_e5_modified":False},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"status":"completed","runs":len(rows),"summary":str((OUT/"summary.json").resolve())},ensure_ascii=False))
if __name__=="__main__":main()
