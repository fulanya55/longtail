"""Compare long-tail losses at the confirmed AMA IF3 BCE bottleneck."""
from __future__ import annotations
import json, itertools, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from cjlt.losses.classification import build_classification_loss
from experiments.run_route_a_c1_baseline import initialize_f2, metrics
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import INIT, set_seed, sha256  # noqa: E402
from experiments.run_route_a_ama_development_v2_20260903 import (SRC_A,SRC_B,OUT as V2_OUT,OLD_ANCHOR,SEL,EXC,load_selected,pool_meta,distribution)

OUT=ROOT/"outputs"/"route_a_ama_longtail_methods_devv2_20260903"
METHODS=("bce","cb_bce","logit_adjustment","drw_cb_bce","positive_focal","asl","combo_sampler","label_combo_joint")
SEEDS=(2039,2040,2041,2042,2043)

class LogitAdjustmentLoss(nn.Module):
    def __init__(self, counts, sample_count, alpha=1.0):
        super().__init__(); prior=torch.as_tensor(counts,dtype=torch.float32)/float(sample_count); self.register_buffer("bias",float(alpha)*torch.log(prior/(1-prior)))
    def forward(self, logits, targets): return F.binary_cross_entropy_with_logits(logits+self.bias.view(1,-1),targets.float())

class LabelComboJointLoss(nn.Module):
    def __init__(self, pair_weight=0.2):
        super().__init__()
        self.pair_weight=float(pair_weight)
    def forward(self, logits, targets):
        bce=F.binary_cross_entropy_with_logits(logits,targets.float()); p=torch.sigmoid(logits); pair=[]
        for i,j in itertools.combinations(range(logits.shape[1]),2):
            pair.append(F.binary_cross_entropy((p[:,i]*p[:,j]).clamp(1e-6,1-1e-6),targets[:,i]*targets[:,j]))
        return bce+self.pair_weight*torch.stack(pair).mean()

def predict(model,signal,labels,device):
    dl=DataLoader(list(zip(torch.from_numpy(signal),torch.from_numpy(labels))),batch_size=64,shuffle=False,num_workers=0,pin_memory=True);ys=[];ls=[];model.eval()
    with torch.no_grad():
        for x,y in dl: ys.append(y.numpy());ls.append(model(x.to(device,non_blocking=True)).cpu().numpy())
    return metrics(np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32))

def train_one(method,seed,fit_signal,fit_labels,val_signal,val_labels,test_signal,test_labels,device):
    set_seed(seed)
    model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=0.2).to(device);init=initialize_f2(model,INIT)
    n=len(fit_labels); counts=fit_labels.astype(np.uint8).sum(0).astype(int)
    if method=="cb_bce": loss_fn=build_classification_loss("cb_bce",positive_counts=counts,beta=0.99)
    elif method=="logit_adjustment": loss_fn=LogitAdjustmentLoss(counts,n)
    elif method=="positive_focal": loss_fn=build_classification_loss("focal",gamma_pos=2.0,gamma_neg=0.0)
    elif method=="asl": loss_fn=build_classification_loss("asl",gamma_pos=0.0,gamma_neg=3.0,clip=0.05)
    elif method=="label_combo_joint": loss_fn=LabelComboJointLoss(pair_weight=0.2)
    else: loss_fn=nn.BCEWithLogitsLoss()
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    cb_loss=build_classification_loss("cb_bce",positive_counts=counts,beta=0.99).to(device) if method=="drw_cb_bce" else None
    ds=list(zip(torch.from_numpy(fit_signal),torch.from_numpy(fit_labels)))
    if method=="combo_sampler":
        codes=fit_labels.astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8)); cc={int(c):int((codes==c).sum()) for c in np.unique(codes)}; w=torch.as_tensor([1.0/cc[int(c)] for c in codes],dtype=torch.double); gen=torch.Generator().manual_seed(seed+70000); sampler=WeightedRandomSampler(w,num_samples=n,replacement=True,generator=gen); loader=DataLoader(ds,batch_size=32,sampler=sampler,num_workers=0,pin_memory=True)
    else: loader=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
    out=OUT/method;out.mkdir(parents=True,exist_ok=True);ck=out/f"{method}__seed_{seed}.pth";opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);hist=[];best=-1.;be=0;start=time.time()
    for ep in range(1,9):
        model.train();ls=[]
        for x,y in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True);lf=cb_loss if method=="drw_cb_bce" and ep>=5 else loss_fn;l=lf(model(x),y);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();ls.append(float(l.detach().cpu()))
        vm=predict(model,val_signal,val_labels,device);hist.append({"epoch":ep,"train_loss":float(np.mean(ls)),"validation":vm});print(json.dumps({"method":method,"seed":seed,"epoch":ep,"val_macro_aupr":vm["macro_aupr"],"val_macro_f1":vm["macro_f1"],"val_combo":vm["combination_exact_accuracy"]}),flush=True)
        if vm["macro_aupr"]>best:best=vm["macro_aupr"];be=ep;torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"validation":vm},ck)
    model.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)["model_state_dict"]);tm=predict(model,test_signal,test_labels,device)
    result={"schema":"route-a-ama-longtail-method-v1","status":"completed","method":method,"seed":seed,"classes":["FMB","AMA","SawFM","SinFM","MFT","ISRJ"],"backbone":"F2_DFMS_Full","loss":method,"optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":32,"epochs":8,"best_epoch":be,"validation":hist[be-1]["validation"],"development_test":tm,"history":hist,"positive_counts":counts.tolist(),"checkpoint_sha256":sha256(ck),"elapsed_seconds":float(time.time()-start),"locked_test_used":False,"frozen_e5_modified":False}
    (out/f"{method}__seed_{seed}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def resume_one(method,seed,fit_labels,val_signal,val_labels,test_signal,test_labels,device):
    ck=OUT/method/f"{method}__seed_{seed}.pth"
    payload=torch.load(str(ck),map_location=device,weights_only=False)
    model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=0.2).to(device)
    model.load_state_dict(payload["model_state_dict"])
    vm=predict(model,val_signal,val_labels,device)
    tm=predict(model,test_signal,test_labels,device)
    return {"schema":"route-a-ama-longtail-method-v1","status":"resumed_from_completed_checkpoint","method":method,"seed":seed,"classes":["FMB","AMA","SawFM","SinFM","MFT","ISRJ"],"backbone":"F2_DFMS_Full","loss":method,"optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":32,"epochs":8,"best_epoch":int(payload["epoch"]),"validation":vm,"development_test":tm,"history":[],"positive_counts":fit_labels.astype(np.uint8).sum(0).astype(int).tolist(),"checkpoint_sha256":sha256(ck),"elapsed_seconds":None,"locked_test_used":False,"frozen_e5_modified":False}

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda:0");OUT.mkdir(parents=True,exist_ok=True)
    m=json.loads((V2_OUT/"manifest.json").read_text(encoding="utf-8"));a_sig,a_lab,_=load_selected(SRC_A,m["anchor_indices"]);v_sig,v_lab,_=load_selected(SRC_A,m["validation_indices"]);t_sig,t_lab,_=load_selected(SRC_A,m["development_test_indices"]);i_sig,i_lab,_=load_selected(SRC_B,m["if3_indices"])
    existing=json.loads((V2_OUT/"summary.json").read_text(encoding="utf-8"));rows=[]
    # Reuse the already completed IF3 baseline exactly; all other methods are trained here.
    for r in existing["runs"]:
        if r["condition"]=="if3": rows.append({"schema":"route-a-ama-longtail-method-v1","status":"reused_completed_baseline","method":"bce","seed":r["seed"],"backbone":r["backbone"],"loss":"BCEWithLogitsLoss","optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch_size":32,"epochs":8,"best_epoch":r["best_epoch"],"validation":r["validation"],"development_test":r["development_test"],"history":r["history"],"checkpoint_sha256":r["checkpoint_sha256"],"locked_test_used":False,"frozen_e5_modified":False})
    for method in METHODS[1:]:
        for seed in SEEDS:
            ck=OUT/method/f"{method}__seed_{seed}.pth"
            receipt=OUT/method/f"{method}__seed_{seed}.json"
            if receipt.exists(): rows.append(json.loads(receipt.read_text(encoding="utf-8")))
            else: rows.append(train_one(method,seed,i_sig,i_lab,v_sig,v_lab,t_sig,t_lab,device))
    summary={"schema":"route-a-ama-longtail-methods-devv2-summary-v1","status":"completed","methods":list(METHODS),"seeds":list(SEEDS),"manifest":m,"common_validation_distribution":distribution(v_lab.astype(np.uint8)),"common_test_distribution":distribution(t_lab.astype(np.uint8)),"runs":rows,"locked_test_used":False,"frozen_e5_modified":False,"evidence_boundary":"development-v2-confirmation_only"}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"locked_test_used":False,"frozen_e5_modified":False},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"status":"completed","summary":str((OUT/"summary.json").resolve()),"runs":len(rows)}),flush=True)
if __name__=="__main__":main()
