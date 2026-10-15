"""P1a train-only mechanism screen for the six-class C1 baseline."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone  # noqa: E402
from experiments.run_route_a_c1_baseline import (  # noqa: E402
    CLASS_NAMES, EXCLUDED_INDICES, SELECTED_INDICES, metrics, pure_six_indices,
)
from experiments.run_route_a_nineclass_feature_sufficiency import (  # noqa: E402
    NineClassDataset, POWERS_DB, find_file, profile_cost, receiver_seeds, sha256, write_json,
)

VARIANTS = {
    "F2_full": "dfms_iq",
    "F2_no_phase": "dfms_iq_no_phase",
    "F2_anti_alias": "dfms_iq_anti_alias",
    "F2_cross_stage": "dfms_iq_csfp",
    "F2_evidence_block": "dfms_iq_epb",
}
SEEDS = (2039, 2040, 2041)


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def initialize_shared(model, checkpoint):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    shared = {
        k[len("backbone."):]: v
        for k, v in payload["model_state_dict"].items()
        if k.startswith("backbone.")
        and not k.startswith("backbone.classifier.")
        and k[len("backbone."):] in model_state
        and tuple(model_state[k[len("backbone."):]].shape) == tuple(v.shape)
    }
    missing, unexpected = model.load_state_dict(shared, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "loaded_shared": len(shared), "missing_non_classifier": missing}


def choose_train_indices(path, split_seed, fit_n, val_n):
    pure = pure_six_indices(path)
    perm = np.random.default_rng(split_seed).permutation(pure)
    fit = np.sort(perm[:fit_n]); val = np.sort(perm[fit_n:fit_n + val_n])
    return fit, val


def predict(model, path, indices, device, batch_size):
    ds = NineClassDataset(path, indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logits, targets = [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                logits.append(model(batch["signal"].to(device, non_blocking=True)).cpu().numpy())
                targets.append(batch["label"].numpy())
    finally:
        ds.close()
    return np.concatenate(targets).astype(np.uint8), np.concatenate(logits).astype(np.float32)


def run_one(args, variant_name, seed, fit_indices, val_indices, power_dev, device, output):
    set_seed(seed)
    model = build_backbone(VARIANTS[variant_name], num_classes=6, width=16, dropout=0.2).to(device)
    init = initialize_shared(model, args.f2_checkpoint_root / f"seed_{seed}" / "backbone" / "best_model.pth")
    train_path = find_file(args.data_root, 30, "train")
    ds = NineClassDataset(train_path, fit_indices, label_indices=np.asarray(SELECTED_INDICES, dtype=np.int64))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(); history=[]; started=time.time(); best=-1.0; best_epoch=0
    try:
        for epoch in range(1, args.epochs+1):
            model.train(); losses=[]
            for batch in loader:
                x=batch["signal"].to(device, non_blocking=True); y=batch["label"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
            yt, yl = predict(model, train_path, val_indices, device, args.eval_batch_size); vm=metrics(yt,yl)
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":vm})
            if vm["macro_aupr"]>best: best=vm["macro_aupr"]; best_epoch=epoch; torch.save(model.state_dict(), output/f"{variant_name}__seed_{seed}.pth")
    finally: ds.close()
    model.load_state_dict(torch.load(str(output/f"{variant_name}__seed_{seed}.pth"), map_location=device, weights_only=True))
    result={"variant":variant_name,"model_name":VARIANTS[variant_name],"seed":seed,"initialization":init,"history":history,"best_epoch":best_epoch,"validation":history[best_epoch-1]["validation"],"power_dev":{},"total_parameters":int(sum(p.numel() for p in model.parameters())),"cost_profile":profile_cost(model,device),"locked_test_used":False,"elapsed_seconds":time.time()-started}
    for power,(path,idx) in power_dev.items():
        yt,yl=predict(model,path,idx,device,args.eval_batch_size); result["power_dev"][str(power)]=metrics(yt,yl)
    result["checkpoint_sha256"]=sha256(output/f"{variant_name}__seed_{seed}.pth")
    write_json(output/f"{variant_name}__seed_{seed}.json",result)
    print(json.dumps({"variant":variant_name,"seed":seed,"val_macro_aupr":result["validation"]["macro_aupr"]},ensure_ascii=False),flush=True)
    return result


def run(args):
    device=torch.device(args.device)
    if device.type!="cuda": raise RuntimeError("P1a requires CUDA")
    variants = [v for v in os.environ.get("P1A_VARIANTS", "").split(",") if v] or list(VARIANTS)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown: raise ValueError(f"unknown P1A_VARIANTS: {unknown}")
    output=args.output_root; output.mkdir(parents=True,exist_ok=True)
    train_path=find_file(args.data_root,30,"train")
    fit,val=choose_train_indices(train_path,args.split_seed,args.fit_samples,args.val_samples)
    power_dev={}
    used=set(receiver_seeds(train_path,fit))
    for power in POWERS_DB:
        path=find_file(args.data_root,power,"train"); pure=pure_six_indices(path)
        with h5py.File(str(path), "r") as handle:
            all_receiver_seeds = np.asarray(handle["receiver_noise_seed"][:, :], dtype=np.int64).reshape(-1)
        candidates = pure[~np.isin(all_receiver_seeds[pure], np.fromiter(used, dtype=np.int64))]
        idx=np.sort(np.random.default_rng(args.split_seed+power).choice(candidates,size=min(args.power_dev_samples,len(candidates)),replace=False)); power_dev[power]=(path,idx)
    role={"schema":"route-a-p1a-train-only-v1","selected_classes":list(CLASS_NAMES),"excluded_classes":["SN","VFT","RangeFakeTarget"],"fit":{"path":str(train_path),"count":len(fit),"sample_indices":fit.tolist(),"receiver_noise_seed":receiver_seeds(train_path,fit)},"validation":{"path":str(train_path),"count":len(val),"sample_indices":val.tolist(),"receiver_noise_seed":receiver_seeds(train_path,val)},"power_development":{"source_role":"train_only_unseen_identity","count_by_power":{str(p):len(i) for p,(path,i) in power_dev.items()},"sample_indices_by_power":{str(p):i.tolist() for p,(path,i) in power_dev.items()}},"test":{"used":False,"locked_test_used":False}}
    write_json(output/"role_manifest.json",role)
    rows=[]
    for name in variants:
        d=output/name; d.mkdir(parents=True,exist_ok=True)
        for seed in args.seeds:
            artifact = d / f"{name}__seed_{seed}.json"
            if artifact.is_file() and args.resume:
                rows.append(json.loads(artifact.read_text(encoding="utf-8")))
            else:
                rows.append(run_one(args,name,int(seed),fit,val,power_dev,device,d))
    summary={"schema":"route-a-p1a-mechanism-summary-v1","status":"completed","locked_test_used":False,"variants":{}}
    for name in variants:
        cells=[r for r in rows if r["variant"]==name]; summary["variants"][name]={"model_name":VARIANTS[name],"parameters":cells[0]["total_parameters"],"val_macro_aupr_mean":float(np.mean([r["validation"]["macro_aupr"] for r in cells])),"val_macro_aupr_std":float(np.std([r["validation"]["macro_aupr"] for r in cells],ddof=1)),"val_combo_mean":float(np.mean([r["validation"]["combination_exact_accuracy"] for r in cells])),"power_dev_macro_aupr_mean_by_power":{str(p):float(np.mean([r["power_dev"][str(p)]["macro_aupr"] for r in cells])) for p in POWERS_DB}}
    write_json(output/"p1a_mechanism_summary.json",summary); write_json(output/"terminal_receipt.json",{"status":"completed","summary":str((output/"p1a_mechanism_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False,"script_sha256":sha256(Path(__file__))})
    print(json.dumps({"status":"completed","summary":str(output/"p1a_mechanism_summary.json")},ensure_ascii=False),flush=True)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,default=Path(r"E:\论文相关\干扰数据\复合干扰数据0716\九类")); p.add_argument("--f2-checkpoint-root",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_visibility_balanced_fusion_v2"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/route_a_p1a_mechanism_screen_20260827"); p.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS)); p.add_argument("--split-seed",type=int,default=260827); p.add_argument("--fit-samples",type=int,default=4096); p.add_argument("--val-samples",type=int,default=2048); p.add_argument("--power-dev-samples",type=int,default=512); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--eval-batch-size",type=int,default=32); p.add_argument("--epochs",type=int,default=2); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--device",default="cuda:0"); p.add_argument("--resume",action="store_true"); return p.parse_args()


if __name__=="__main__": run(parse_args())
