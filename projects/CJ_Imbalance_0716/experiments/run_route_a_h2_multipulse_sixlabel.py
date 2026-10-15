"""Train auditable C0--C4 H2 multi-pulse arms on role-isolated sources.

``--smoke`` is deliberately marked non-evidence and accepts a single source
for both loader roles only to validate the engineering path.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cjlt.models.multipulse_vft_expert import MODES, MultiPulseVFTExpertClassifier


class H5SixLabelDataset(Dataset):
    def __init__(self, path: Path):
        self.path = Path(path); self.handle = None
        with h5py.File(self.path, "r") as h5: self.length = int(h5["labels"].shape[0])
    def _file(self):
        if self.handle is None: self.handle = h5py.File(self.path, "r")
        return self.handle
    def __len__(self): return self.length
    def __getitem__(self, index):
        h5 = self._file(); iq = np.stack((h5["iq_real"][index], h5["iq_imag"][index]), axis=0).astype(np.float32, copy=False)
        return {"iq": torch.from_numpy(iq), "labels": torch.from_numpy(np.asarray(h5["labels"][index], dtype=np.float32)), "power": torch.from_numpy(np.asarray(h5["component_jnr_db"][index], dtype=np.float32)), "pair_seed": int(h5["pair_seed"][index])}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval(); logits=[]; labels=[]; power=[]; pair=[]
    with torch.inference_mode():
        for batch in loader:
            logits.append(model(batch["iq"].to(device)).cpu().numpy()); labels.append(batch["labels"].numpy()); power.append(batch["power"].numpy()); pair.extend(batch["pair_seed"].tolist())
    raw_logits=np.concatenate(logits); score = 1/(1+np.exp(-raw_logits)); target=np.concatenate(labels); component_power=np.concatenate(power)
    per_label=[]
    for i in range(6):
        y=target[:,i]; p=score[:,i]; per_label.append({"aupr":float(average_precision_score(y,p)) if np.unique(y).size == 2 else float("nan"),"auroc":float(roc_auc_score(y,p)) if np.unique(y).size == 2 else float("nan")})
    weak=(target[:,4] == 1) & np.isfinite(component_power[:,4]) & (component_power[:,4] <= 8.001); negatives=target[:,4] == 0; selected=weak | negatives
    vft_weak_auroc=float(roc_auc_score(target[selected,4],score[selected,4])) if weak.any() and negatives.any() else float("nan")
    return {"macro_aupr":float(np.nanmean([r["aupr"] for r in per_label])),"per_label":per_label,"vft_weak_auroc":vft_weak_auroc,"logits_raw":raw_logits,"predictions":score,"labels":target,"power":component_power,"pair_seed":np.asarray(pair,dtype=np.uint64)}


def _fit_temperatures(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fit one temperature per label using only the calibration pair half."""
    temperatures=[]
    for index in range(labels.shape[1]):
        x=torch.tensor(logits[mask,index],dtype=torch.float32); y=torch.tensor(labels[mask,index],dtype=torch.float32)
        value=torch.zeros((),requires_grad=True); optimizer=torch.optim.Adam([value],lr=.05)
        for _ in range(100):
            optimizer.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(x/torch.exp(value).clamp(.05,20),y); loss.backward();optimizer.step()
        temperatures.append(float(torch.exp(value.detach()).clamp(.05,20)))
    return np.asarray(temperatures,dtype=np.float64)


def _select_thresholds(probability: np.ndarray, labels: np.ndarray, power: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Use threshold half only; VFT maximizes weak recall at FPR <= 0.15."""
    thresholds=[]
    for index in range(labels.shape[1]):
        y=labels[mask,index].astype(bool); score=probability[mask,index]; candidates=np.unique(np.quantile(score,np.linspace(0,1,201)))
        if index == 4:
            weak=y & np.isfinite(power[mask,index]) & (power[mask,index] <= 8.001); negative=~y; valid=[]
            for candidate in candidates:
                fpr=float((score[negative] >= candidate).mean()) if negative.any() else 0.0; recall=float((score[weak] >= candidate).mean()) if weak.any() else 0.0
                if fpr <= .15: valid.append((recall,-fpr,float(candidate)))
            thresholds.append(max(valid)[2] if valid else 1.0)
        else:
            values=[]
            for candidate in candidates:
                prediction=score>=candidate; tp=np.sum(prediction & y); fp=np.sum(prediction & ~y); fn=np.sum(~prediction & y); f1=(2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.; values.append((f1,float(candidate)))
            thresholds.append(max(values)[1])
    return np.asarray(thresholds,dtype=np.float64)


def _threshold_metrics(probability: np.ndarray, labels: np.ndarray, power: np.ndarray, threshold: np.ndarray) -> dict[str, object]:
    per=[]
    for index in range(labels.shape[1]):
        y=labels[:,index].astype(bool); pred=probability[:,index]>=threshold[index]; negative=~y; fpr=float((pred[negative]).mean()) if negative.any() else float("nan")
        weak=y & np.isfinite(power[:,index]) & (power[:,index] <= 8.001); weak_recall=float(pred[weak].mean()) if weak.any() else float("nan")
        per.append({"threshold":float(threshold[index]),"fpr":fpr,"weak_recall":weak_recall})
    return {"per_label":per}


def run_arm(mode: str, train_path: Path, dev_path: Path, *, seed: int, epochs: int, lr: float, batch_size: int, width: int, expert_dim: int, device: torch.device, output: Path) -> dict[str, object]:
    seed_everything(seed); train=H5SixLabelDataset(train_path); dev=H5SixLabelDataset(dev_path); train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0); dev_loader=DataLoader(dev,batch_size=batch_size,shuffle=False,num_workers=0)
    model=MultiPulseVFTExpertClassifier(mode,width=width,expert_dim=expert_dim).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); loss_fn=nn.BCEWithLogitsLoss(); best=None; history=[]; started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); running=0.; count=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); logits=model(batch["iq"].to(device)); loss=loss_fn(logits,batch["labels"].to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); running += float(loss.detach())*len(logits); count += len(logits)
        result=evaluate(model,dev_loader,device); row={"epoch":epoch,"train_bce":running/max(count,1),"macro_aupr":result["macro_aupr"],"vft_weak_auroc":result["vft_weak_auroc"]}; history.append(row)
        print(json.dumps({"event": "epoch_complete", "mode": mode, "seed": seed, **row}), flush=True)
        if best is None or row["macro_aupr"] > best["metrics"]["macro_aupr"]:
            best={"epoch":epoch,"metrics":result,"state":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()}}
    model.load_state_dict(best.pop("state")); output.mkdir(parents=True,exist_ok=True); checkpoint=output/"checkpoint.pt"; torch.save({"mode":mode,"seed":seed,"state_dict":model.state_dict(),"best_epoch":best["epoch"],"history":history},checkpoint)
    metric=best["metrics"]; calibration_mask=(metric["pair_seed"] % 2) == 0; threshold_mask=~calibration_mask
    temperatures=_fit_temperatures(metric["logits_raw"],metric["labels"],calibration_mask); calibrated=1/(1+np.exp(-metric["logits_raw"]/temperatures[None,:])); thresholds=_select_thresholds(calibrated,metric["labels"],metric["power"],threshold_mask); threshold_metrics=_threshold_metrics(calibrated,metric["labels"],metric["power"],thresholds)
    np.savez_compressed(output/"development_predictions.npz",raw_logits=metric.pop("logits_raw"),probabilities=metric.pop("predictions"),calibrated_probabilities=calibrated,labels=metric.pop("labels"),component_jnr_db=metric.pop("power"),pair_seed=metric.pop("pair_seed"))
    (output/"calibration_and_thresholds.json").write_text(json.dumps({"pair_split":"pair_seed mod 2; even=calibration, odd=threshold selection","temperature":temperatures.tolist(),"threshold":thresholds.tolist(),"threshold_metrics":threshold_metrics},indent=2)+"\n",encoding="utf-8")
    return {"mode":mode,"seed":seed,"best_epoch":best["epoch"],"history":history,"metrics":metric,"threshold_metrics":threshold_metrics,"temperature":temperatures.tolist(),"threshold":thresholds.tolist(),"checkpoint":str(checkpoint),"parameter_count":sum(p.numel() for p in model.parameters()),"train_seconds":time.perf_counter()-started}


def main(args: argparse.Namespace) -> dict[str, object]:
    if not args.smoke and args.train_path.resolve() == args.development_path.resolve(): raise ValueError("formal train/development paths must differ")
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); root=args.output_root; results=[]
    for seed in args.seeds:
        for mode in args.modes:
            results.append(run_arm(mode,args.train_path,args.development_path,seed=seed,epochs=args.epochs,lr=args.lr,batch_size=args.batch_size,width=args.width,expert_dim=args.expert_dim,device=device,output=root/f"{mode}_seed{seed}"))
    receipt={"status":"completed_smoke_only" if args.smoke else "completed_development_unaggregated","experiment":"RouteA_H2_MultiPulseSixLabel","modes":args.modes,"seeds":args.seeds,"source":{"train":str(args.train_path),"development":str(args.development_path)},"config":{"epochs":args.epochs,"lr":args.lr,"batch_size":args.batch_size,"width":args.width,"expert_dim":args.expert_dim,"loss":"BCEWithLogitsLoss","optimizer":"AdamW"},"device":str(device),"results":results,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"smoke":args.smoke}
    root.mkdir(parents=True,exist_ok=True); (root/"run_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return receipt


if __name__ == "__main__":
    smoke=ROOT / ".." / ".."  # argument defaults are resolved below
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train-path",type=Path,required=True);p.add_argument("--development-path",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--modes",nargs="+",choices=tuple(MODES),default=list(MODES));p.add_argument("--seeds",nargs="+",type=int,default=[20260803]);p.add_argument("--epochs",type=int,default=24);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--width",type=int,default=32);p.add_argument("--expert-dim",type=int,default=16);p.add_argument("--device",default=None);p.add_argument("--smoke",action="store_true"); a=p.parse_args(); value=main(a);print(json.dumps({"status":value["status"],"modes":value["modes"],"seeds":value["seeds"]},ensure_ascii=False))
