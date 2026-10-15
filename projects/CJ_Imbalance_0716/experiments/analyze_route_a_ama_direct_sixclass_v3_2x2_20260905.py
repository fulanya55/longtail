"""Calibration-only analysis for direct six-class development-v3 2x2 runs.

The calibration split is the only split used to fit thresholds/temperature.  The
check split is a confirmation split and development-test is evaluated once with
the frozen calibration decisions.  No locked-test artifact is read.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from run_route_a_ama_direct_sixclass_v3_2x2_20260905 import (
    DATA, OUT, SEEDS, CLASS_NAMES, DS, load, sha256, build_backbone, INIT
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GRID = np.linspace(0.05, 0.95, 181, dtype=np.float64)


def probs(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))


def ece(y, p, bins=15):
    y = y.astype(np.float64); p = p.astype(np.float64)
    vals = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any(): vals.append(float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean())))
    return float(sum(vals))


def evaluate(y, logits, thresholds=None, temperature=1.0, train_codes=None):
    p = probs(logits / float(temperature))
    t = np.full(6, 0.5, dtype=np.float64) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    pred = (p >= t.reshape(1, -1)).astype(np.uint8)
    powers = (2 ** np.arange(6, dtype=np.int64))[None, :]
    tc = (y.astype(np.int64) * powers).sum(1); pc = (pred.astype(np.int64) * powers).sum(1)
    per = {}
    auprs=[]; aurocs=[]; eces=[]; briers=[]
    for i, name in enumerate(CLASS_NAMES):
        truth = y[:, i].astype(np.uint8); positive = truth == 1; negative = ~positive
        pp = p[:, i]
        auprs.append(float(average_precision_score(truth, pp)))
        aurocs.append(float(roc_auc_score(truth, pp)) if len(np.unique(truth)) == 2 else float("nan"))
        eces.append(ece(truth, pp)); briers.append(float(np.mean((pp-truth)**2)))
        per[name] = {"recall": float(pred[positive, i].mean()), "negative_fpr": float(pred[negative, i].mean()),
                     "f1": float(f1_score(truth, pred[:, i], zero_division=0)), "positive_count": int(positive.sum()),
                     "ece": eces[-1], "brier": briers[-1]}
    out = {"thresholds": t.tolist(), "temperature": float(temperature),
           "macro_aupr": float(np.nanmean(auprs)), "macro_auroc": float(np.nanmean(aurocs)),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "hamming_loss": float(np.mean(pred != y)), "combination_exact_accuracy": float(np.mean(tc == pc)),
           "subset_accuracy": float(np.mean(np.all(pred == y, axis=1))), "ece": float(np.mean(eces)),
           "brier": float(np.mean(briers)), "per_label": per}
    if train_codes is not None:
        counts = np.bincount(train_codes.astype(int), minlength=64)[1:]
        order = np.argsort(counts)
        rare = (order[:13] + 1).astype(np.int64); common = (order[-13:] + 1).astype(np.int64)
        out["rare_combination_codes"] = rare.tolist(); out["common_combination_codes"] = common.tolist()
        out["rare_combination_accuracy"] = float(np.mean(tc[np.isin(tc, rare)] == pc[np.isin(tc, rare)]))
        out["common_combination_accuracy"] = float(np.mean(tc[np.isin(tc, common)] == pc[np.isin(tc, common)]))
        out["rare_combination_count"] = int(np.isin(tc, rare).sum()); out["common_combination_count"] = int(np.isin(tc, common).sum())
    return out


def fit_f1_thresholds(y, p):
    ts=[]
    for i in range(6):
        scores=[f1_score(y[:, i], p[:, i] >= t, zero_division=0) for t in GRID]
        ts.append(float(GRID[int(np.argmax(scores))]))
    return np.asarray(ts)


def fit_combo_thresholds(y, p, start):
    t = np.asarray(start, dtype=np.float64).copy(); powers=(2**np.arange(6,dtype=np.int64))[None,:]; tc=(y.astype(np.int64)*powers).sum(1)
    def score(tt): return float(np.mean(tc == ((p >= tt.reshape(1,-1)).astype(np.int64)*powers).sum(1)))
    best=score(t)
    for _ in range(2):
        for i in range(6):
            cur=t[i]; vals=[]
            for v in GRID:
                q=t.copy(); q[i]=v; vals.append(score(q))
            j=int(np.argmax(vals)); t[i]=float(GRID[j]); best=max(best,float(vals[j]))
    return t, float(best)


def fit_temperature(y, logits):
    # Scalar temperature chosen by calibration BCE; threshold remains 0.5.
    best=(float("inf"),1.0)
    for temp in np.linspace(0.5,3.0,101):
        z=np.clip(logits.astype(np.float64)/temp,-40,40); loss=np.mean(np.logaddexp(0,z)-y*z)
        if loss < best[0]: best=(float(loss),float(temp))
    return best[1]


def infer(ck, x, y):
    model=build_backbone("dfms_iq",num_classes=6,width=16,dropout=.2).to(DEVICE)
    payload=torch.load(str(ck),map_location=DEVICE,weights_only=False); model.load_state_dict(payload["model_state_dict"]); model.eval()
    out=[]; t0=time.time()
    with torch.no_grad():
        for xb,_ in DataLoader(DS(x,y),batch_size=64,shuffle=False,num_workers=0,pin_memory=True): out.append(model(xb.to(DEVICE,non_blocking=True)).cpu().numpy())
    return np.concatenate(out).astype(np.float32), float(time.time()-t0), sum(p.numel() for p in model.parameters())


def paired_ci(x, base, seed=20260905, reps=10000):
    d=np.asarray(x,float)-np.asarray(base,float); rng=np.random.default_rng(seed); idx=rng.integers(0,len(d),size=(reps,len(d))); means=d[idx].mean(1)
    return float(d.mean()), [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def main():
    (OUT/"analysis").mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8")); summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    _, ytr, _=load(DATA/"closedset_6class_train.mat", manifest["fit_indices"]); train_codes=(ytr.astype(np.uint8)@(2**np.arange(6,dtype=np.uint8))).astype(np.int64)
    splits={}
    for name,key in (("calibration","calibration_indices"),("check","check_indices"),("development_test","development_test_indices")):
        x,y,_=load(DATA/"closedset_6class_test.mat", manifest[key]); splits[name]=(x,y)
    rows=[]; start=time.time()
    for run in summary["runs"]:
        cond=run["condition"]; seed=run["seed"]; ck=OUT/cond/f"{cond}__seed_{seed}.pth"
        logits={}; infer_meta={}
        for name,(x,y) in splits.items(): logits[name],infer_meta[name],params=infer(ck,x,y)
        yc=splits["calibration"][1]; pc=probs(logits["calibration"]); tf1=fit_f1_thresholds(yc,pc); tcombo,combo_cal=fit_combo_thresholds(yc,pc,tf1); temp=fit_temperature(yc,logits["calibration"])
        decisions={"fixed_0.5":{"thresholds":np.full(6,.5),"temperature":1.0},"f1_calibrated":{"thresholds":tf1,"temperature":1.0},"combo_calibrated":{"thresholds":tcombo,"temperature":1.0},"temperature_calibrated":{"thresholds":np.full(6,.5),"temperature":temp}}
        metrics_by={}
        for split,(x,y) in splits.items():
            metrics_by[split]={name:evaluate(y,logits[split],cfg["thresholds"],cfg["temperature"],train_codes) for name,cfg in decisions.items()}
        row={"condition":cond,"seed":int(seed),"threshold_fit_split":"calibration","thresholds":{"f1":tf1.tolist(),"combo":tcombo.tolist(),"temperature":float(temp)},"calibration_combo_objective":combo_cal,"metrics":metrics_by,"inference_seconds":infer_meta,"parameter_count":int(params),"locked_test_used":False,"frozen_e5_modified":False}
        rows.append(row); (OUT/"analysis"/f"{cond}__seed_{seed}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
    aggregate={}
    for split in ("calibration","check","development_test"):
        aggregate[split]={}
        for decision in ("fixed_0.5","f1_calibrated","combo_calibrated","temperature_calibrated"):
            aggregate[split][decision]={}
            for cond in summary["conditions"]:
                rs=[r["metrics"][split][decision] for r in rows if r["condition"]==cond]
                aggregate[split][decision][cond]={k:{"mean":float(np.mean([m[k] for m in rs])),"sd":float(np.std([m[k] for m in rs],ddof=1))} for k in ("macro_aupr","macro_f1","combination_exact_accuracy","hamming_loss","ece","brier","rare_combination_accuracy","common_combination_accuracy")}
            base=[r["metrics"][split][decision] for r in rows if r["condition"]=="bce_ordinary"]
            for cond in summary["conditions"]:
                if cond=="bce_ordinary": continue
                cur=[r["metrics"][split][decision] for r in rows if r["condition"]==cond]
                aggregate[split][decision][cond]["delta_vs_bce"]={k:paired_ci([m[k] for m in cur],[m[k] for m in base]) for k in ("macro_f1","combination_exact_accuracy","hamming_loss","ece","rare_combination_accuracy")}
    result={"schema":"route-a-ama-direct-sixclass-v3-calibration-analysis-v1","status":"completed","dataset":str(DATA),"runs":rows,"aggregate":aggregate,"locked_test_used":False,"frozen_e5_modified":False,"evidence_boundary":"fresh-sixclass-development-v3; calibration-fitted decisions confirmed on check and frozen on development-test","elapsed_seconds":float(time.time()-start)}
    (OUT/"analysis_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"status":"completed","summary":str((OUT/"analysis_summary.json").resolve()),"runs":len(rows)},ensure_ascii=False))

if __name__=="__main__": main()
