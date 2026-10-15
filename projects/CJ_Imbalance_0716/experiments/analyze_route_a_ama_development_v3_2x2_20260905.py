"""Development-v3 analysis: calibration/check separation, composition errors and gates."""
from __future__ import annotations
import json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import brier_score_loss
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_ama_development_v3_2x2_20260905 import SRC_TRAIN,SRC_TEST,OUT,load,dist
from experiments.run_route_a_c1_baseline import metrics
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import sha256
SEEDS=(2050,2051,2052,2053,2054); METHODS=("bce_ordinary","logit_ordinary","bce_balanced","logit_balanced")
class DS(Dataset):
    def __init__(self,x,y):self.x=x;self.y=y
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return torch.from_numpy(self.x[i]),torch.from_numpy(self.y[i])
def infer(ck,x,device):
    m=build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).to(device);m.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)["model_state_dict"]);m.eval();o=[]
    with torch.no_grad():
        for xb,_ in DataLoader(DS(x,np.zeros((len(x),6),np.float32)),batch_size=128,shuffle=False):o.append(m(xb.to(device)).cpu().numpy())
    return np.concatenate(o)
def ece(logits,y,bins=15):
    p=1/(1+np.exp(-np.clip(logits,-40,40)));out=[]
    for j in range(6):
        z=p[:,j]; t=y[:,j]; e=0
        for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(0,1,bins+1)):
            q=(z>=lo)&(z<hi if hi<1 else z<=hi)
            if q.any():e+=q.mean()*abs(z[q].mean()-t[q].mean())
        out.append(e)
    return float(np.mean(out))
def threshold_fit(logits,y):
    p=1/(1+np.exp(-np.clip(logits,-40,40))); ts=[]
    for j in range(6):
        best=(.5,-1)
        for t in np.unique(np.r_[np.linspace(.05,.95,37),p[:,j]]):
            z=(p[:,j]>=t).astype(np.uint8); tp=((z==1)&(y[:,j]==1)).sum(); fp=((z==1)&(y[:,j]==0)).sum(); fn=((z==0)&(y[:,j]==1)).sum(); f=2*tp/max(2*tp+fp+fn,1)
            if f>best[1]:best=(float(t),float(f))
        ts.append(best[0])
    return np.asarray(ts)
def apply_threshold(logits,t):
    z=logits-np.log(np.asarray(t)/(1-np.asarray(t))).reshape(1,-1);return z
def paired(values):
    x=np.asarray(values,float); return {"mean":float(x.mean()),"std":float(x.std(ddof=1)),"positive_seeds":int((x>0).sum()),"wilcoxon_p":float(wilcoxon(x,method="exact").pvalue) if np.any(x) else 1.0}
def main():
    d=json.loads((OUT/"summary.json").read_text(encoding="utf-8"));m=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"));fx,fy,_=load(SRC_TRAIN,m["fit_indices"]);cx,cy,_=load(SRC_TEST,m["calibration_indices"]);qx,qy,_=load(SRC_TEST,m["check_indices"]);tx,ty,_=load(SRC_TEST,m["development_test_indices"]);device=torch.device("cuda:0")
    fit_codes=fy.astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8)); cc=Counter(fit_codes.tolist()); rare=[k for k,_ in sorted(cc.items(),key=lambda z:(z[1],z[0]))[:13]]; rows=[]
    for cond in METHODS:
        for seed in SEEDS:
            ck=OUT/cond/f"{cond}__seed_{seed}.pth"; lc=infer(ck,cx,device); lq=infer(ck,qx,device); lt=infer(ck,tx,device); th=threshold_fit(lc,cy)
            r={"condition":cond,"seed":seed,"thresholds":th.tolist(),"calibration_ece":ece(lc,cy),"calibration_brier":float(np.mean([brier_score_loss(cy[:,j],1/(1+np.exp(-np.clip(lc[:,j],-40,40)))) for j in range(6)]))}
            for name,z,y in (("check",lq,qy),("development_test",lt,ty)):
                for variant,w in (("fixed_0.5",z),("per_class_calibrated",apply_threshold(z,th))):
                    q=metrics(y.astype(np.uint8),w); codes_y=y.astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8));codes_p=(1/(1+np.exp(-np.clip(w,-40,40)))>=.5).astype(np.uint8)@(1<<np.arange(6,dtype=np.uint8)); rare_m=np.isin(codes_y,rare); tail=1; head=0; pp=(1/(1+np.exp(-np.clip(w,-40,40)))>=.5).astype(np.uint8); r[f"{name}_{variant}"]={"macro_aupr":q["macro_aupr"],"macro_f1":q["macro_f1"],"combination_exact_accuracy":q["combination_exact_accuracy"],"hamming_loss":q["hamming_loss"],"tail_AMA_recall":float(pp[y[:,tail]==1,tail].mean()),"head_FMB_fpr":float(pp[y[:,head]==0,head].mean()),"rare_combination_accuracy":float((codes_p[rare_m]==codes_y[rare_m]).mean()),"ece":ece(w,y)}
            rows.append(r)
    g=defaultdict(list);[g[r["condition"]].append(r) for r in rows]; agg={}; gates={};base={r["seed"]:r for r in g["bce_ordinary"]}
    for cond,rs in g.items():
        agg[cond]={}
        for split in ("check","development_test"):
            agg[cond][split]={}
            for variant in ("fixed_0.5","per_class_calibrated"):
                agg[cond][split][variant]={}
                for k in ("macro_aupr","macro_f1","combination_exact_accuracy","hamming_loss","tail_AMA_recall","head_FMB_fpr","rare_combination_accuracy","ece"):
                    v=np.asarray([r[f"{split}_{variant}"][k] for r in rs]);agg[cond][split][variant][k]={"mean":float(v.mean()),"std":float(v.std(ddof=1))}
        agg[cond]["global_repetition_rate"]=float(np.mean([json.loads((OUT/cond/f"{cond}__seed_{r['seed']}.json").read_text(encoding='utf-8'))["global_repetition_rate"] for r in rs]))
        if cond=="bce_ordinary":continue
        dd={}; split="development_test";variant="per_class_calibrated"
        for k in ("macro_aupr","macro_f1","combination_exact_accuracy","hamming_loss","tail_AMA_recall","head_FMB_fpr","rare_combination_accuracy"):
            vals=np.asarray([r[f"{split}_{variant}"][k]-base[r["seed"]][f"{split}_{variant}"][k] for r in rs]);dd[k]=paired(vals)
        checks={"tail_ge_3pp":dd["tail_AMA_recall"]["mean"]>=.03,"rare_ge_2pp":dd["rare_combination_accuracy"]["mean"]>=.02,"f1_non_decrease":dd["macro_f1"]["mean"]>=0,"aupr_ge_minus_0.1pp":dd["macro_aupr"]["mean"]>=-.001,"head_fpr_increase_le_0.5pp":dd["head_FMB_fpr"]["mean"]<=.005,"combo_non_decrease":dd["combination_exact_accuracy"]["mean"]>=0,"tail_4of5":dd["tail_AMA_recall"]["positive_seeds"]>=4,"rare_4of5":dd["rare_combination_accuracy"]["positive_seeds"]>=4};gates[cond]={"delta":dd,"checks":checks,"promote":all(checks.values())}
    out={"schema":"route-a-ama-development-v3-2x2-analysis-v1","status":"completed","rare_codes":rare,"rare_fit_counts":{str(k):cc[k] for k in rare},"tail_label":"AMA","head_label":"FMB","rows":rows,"aggregate":agg,"promotion_gates":gates,"globally_unique_balance_feasible":m["no_replacement_balance_audit"]["globally_unique_fixed_budget_feasible"],"evidence_boundary":"development-v3-screening","locked_test_used":False,"frozen_e5_modified":False}
    p=OUT/"analysis.json";p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"status":"completed","output":str(p.resolve()),"promoted":[c for c,g in gates.items() if g["promote"]]},ensure_ascii=False))
if __name__=="__main__":main()
