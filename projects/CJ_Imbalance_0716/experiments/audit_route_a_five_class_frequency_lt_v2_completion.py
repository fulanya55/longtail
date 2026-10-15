"""Complete post-run diagnostics without model or hyperparameter reselection."""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from experiments.run_route_a_five_class_frequency_lt_v2 import (
    OUT_ROOT, AUDIT_ROOT, TRAIN_PATH, CLASS_NAMES, SEEDS, ensure_stage0,
    build_criterion, logits_from_checkpoint, build_backbone, DEVICE,
)


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tiers(counts):
    ordered=sorted(counts,key=lambda k:(-counts[k],k)); chunks=np.array_split(np.asarray(ordered,dtype=object),3)
    return {str(k): tier for tier,vals in zip(("head","middle","tail"),chunks) for k in vals.tolist()}


def main():
    data=ensure_stage0(); labels=data["train5"]
    s1=json.loads((OUT_ROOT/"stage1_summary.json").read_text(encoding="utf-8")); s2=json.loads((OUT_ROOT/"stage2_summary.json").read_text(encoding="utf-8"))
    # Co-occurrence and sampler exposure diagnostics.
    sampler_diag={}
    for seed in SEEDS:
        roles={k:np.asarray(v,dtype=np.int64) for k,v in json.loads((OUT_ROOT/"splits"/f"seed_{seed}.json").read_text(encoding="utf-8")).items()}
        y=labels[roles["train"]]; counts=y.sum(0).astype(float); inv=1/np.maximum(counts,1)
        sampler_diag[str(seed)]={}
        for name in ("uniform","max_positive","mean_positive"):
            if name=="uniform": w=np.ones(len(y))
            else:
                per=y*inv[None,:]
                w=np.max(np.where(y>0,per,0),axis=1) if name=="max_positive" else np.sum(per,axis=1)/np.maximum(y.sum(1),1)
            p=w/w.sum(); rng=np.random.RandomState(seed+{"uniform":17,"max_positive":18,"mean_positive":19}[name]); pos=rng.choice(len(y),size=9920,replace=True,p=p); sampled=y[pos]
            sampler_diag[str(seed)][name]={"draw_count":9920,"unique_identity_count":int(len(np.unique(pos))),"repeat_fraction":float(1-len(np.unique(pos))/9920),"effective_label_exposure":dict(zip(CLASS_NAMES,sampled.sum(0).astype(int).tolist())),"sample_weight_min":float(w.min()),"sample_weight_max":float(w.max()),"sample_weight_cv":float(w.std()/max(w.mean(),1e-12))}
    balanced=data["audit"]["balanced_combo_counts"]; longtail=data["audit"]["label_if4_combo_counts"]
    def cooc(indices):
        y=labels[indices]; mat=y.T@y
        return {"matrix":mat.astype(int).tolist(),"partner_counts":{CLASS_NAMES[i]:int(np.sum((mat[i]>0)&(np.arange(5)!=i))) for i in range(5)}}
    coverage={"balanced":{"combo_counts":balanced,"combo_tiers":tiers(balanced),**cooc(data["train_valid"])},"label_if4":{"combo_counts":longtail,"combo_tiers":tiers(longtail),**cooc(data["manifest_indices"])},"sampler_diagnostics":sampler_diag}
    dump(OUT_ROOT/"sampler_and_cooccurrence_diagnostics.json",coverage)

    # Per-label positive/negative logit-gradient magnitude proxy on frozen early predictions.
    grad={}
    cfg=dict((name,cfg) for name,cfg in __import__('experiments.run_route_a_five_class_frequency_lt_v2',fromlist=['STAGE2_LOSSES']).STAGE2_LOSSES)
    for arm,c in cfg.items():
        grad[arm]=[]
        for row in [r for r in s2["rows"] if r["arm"]==arm]:
            p=np.load(Path(row["checkpoint"]).parent/"early_predictions.npz"); logits=torch.tensor(p["logits"],dtype=torch.float32,requires_grad=True); targets=torch.tensor(p["targets"],dtype=torch.float32)
            crit=build_criterion(c["kind"],p["targets"],c); loss=crit(logits,targets); loss.backward(); g=logits.grad.detach().abs().numpy(); y=p["targets"]
            grad[arm].append({"seed":row["seed"],"per_label":{CLASS_NAMES[i]:{"positive_mean_abs_dloss_dlogit":float(g[y[:,i]==1,i].mean()),"negative_mean_abs_dloss_dlogit":float(g[y[:,i]==0,i].mean())} for i in range(5)}})
    dump(OUT_ROOT/"gradient_proxy_analysis.json",{"definition":"mean absolute d(loss)/d(logit) recomputed on saved early-role logits; diagnostic proxy, not a stored train-batch gradient trace","arms":grad})

    # Engineering benchmark for the fixed B0 capacity.
    selected=next(r for r in s2["rows"] if r["arm"]==s2["selected_loss"] and r["seed"]==SEEDS[0]); model=build_backbone("dfms_iq",num_classes=5,width=16,dropout=.2).to(DEVICE); model.load_state_dict(torch.load(selected["checkpoint"],map_location="cpu")); model.eval()
    x=torch.randn(64,2,3000,device=DEVICE); torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    with torch.no_grad():
        for _ in range(5): model(x)
        if DEVICE.type=="cuda": torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(30): model(x)
        if DEVICE.type=="cuda": torch.cuda.synchronize()
    elapsed=time.perf_counter()-start; params=sum(p.numel() for p in model.parameters())
    training_times={arm:[float(json.loads(Path(r["history"]).read_text(encoding="utf-8"))[-1]["elapsed_seconds"]) for r in s2["rows"] if r["arm"]==arm] for arm in cfg}
    eng={"device":str(DEVICE),"batch_size":64,"iterations":30,"samples":1920,"elapsed_seconds":elapsed,"throughput_samples_per_second":1920/elapsed,"latency_ms_per_sample":1000*elapsed/1920,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()) if DEVICE.type=="cuda" else None,"parameter_count":int(params),"training_elapsed_seconds_by_arm":training_times}
    dump(OUT_ROOT/"engineering_metrics.json",eng)
    dump(OUT_ROOT/"completion_audit_receipt.json",{"status":"completed","model_reselection":False,"test_reselection":False,"artifacts":["sampler_and_cooccurrence_diagnostics.json","gradient_proxy_analysis.json","engineering_metrics.json"],"generated_at_unix":time.time()})
    print("COMPLETION_AUDIT_OK")

if __name__=="__main__": main()
