"""Train/evaluate C1 BCE on fixed-budget IF-sweep manifests."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import (CLASS_NAMES, SELECTED_INDICES, NineClassDataset,
                                                 initialize_f2, metrics, predict)
from experiments.run_route_a_nineclass_feature_sufficiency import find_file, profile_cost, POWERS_DB

ROLE = ROOT / "outputs/route_a_c1_baseline_full_20260826/role_manifest.json"
INIT = Path(r"E:\LJQ_code\CJ_Imbalance_0716\frozen\20260731_route_a_visibility_balanced_fusion_v2\seed_2039\backbone\best_model.pth")
MANIFEST_ROOT = ROOT / "outputs/route_a_if_sweep_20260902/manifests"
OUT = ROOT / "outputs/route_a_if_sweep_20260902/runs"


class CachedDataset(Dataset):
    """In-memory normalized I/Q arrays; avoids repeated HDF5 scalar reads."""
    def __init__(self, path: Path, indices: np.ndarray):
        with __import__('h5py').File(str(path), 'r') as h:
            labels = np.asarray(h['known_label'][list(SELECTED_INDICES), :][:, indices], dtype=np.uint8).T
            stored = np.asarray(h['iq_complex'][:, indices])
        if stored.dtype.fields is None: raise TypeError('iq_complex must be compound real/imag')
        signal = np.stack([np.asarray(stored['real'], dtype=np.float32).T,
                           np.asarray(stored['imag'], dtype=np.float32).T], axis=1)
        rms = np.sqrt(np.mean(signal[:,0]**2 + signal[:,1]**2, axis=1))
        signal = signal / np.sqrt(rms[:,None,None]**2 + 1e-12)
        self.signal = np.ascontiguousarray(signal, dtype=np.float32)
        self.labels = labels.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        return {'signal':torch.from_numpy(self.signal[i]),'label':torch.from_numpy(self.labels[i]),'sample_index':int(self.indices[i])}


def predict_cached(model, ds, device, batch_size):
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True); ys=[]; ls=[]; model.eval()
    with torch.no_grad():
        for b in dl:
            ls.append(model(b['signal'].to(device,non_blocking=True)).cpu().numpy()); ys.append(b['label'].numpy())
    return np.concatenate(ys).astype(np.uint8), np.concatenate(ls).astype(np.float32)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--conditions',nargs='+',default=['anchor_IF1','IF6','IF10','IF20']); ap.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041]); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--test-samples',type=int,default=2048); ap.add_argument('--full-test',action='store_true'); ap.add_argument('--skip-test',action='store_true'); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device(args.device); OUT.mkdir(parents=True,exist_ok=True)
    role=json.loads(ROLE.read_text(encoding='utf-8')); val=np.asarray(role['validation']['sample_indices'],dtype=np.int64)
    test_paths={int(k):Path(v) for k,v in role['test']['paths_by_power'].items()}; test_indices={}
    for k,v in role['test']['sample_indices_by_power'].items():
        power=int(k); idx=np.asarray(v,dtype=np.int64)
        if not args.skip_test and not args.full_test and len(idx)>args.test_samples:
            idx=np.sort(np.random.default_rng(260826+power).choice(idx,size=args.test_samples,replace=False))
        test_indices[power]=idx
    all_rows=[]
    for condition in args.conditions:
        for seed in args.seeds:
            mp=MANIFEST_ROOT/f'{condition}__seed_{seed}.json'; manifest=json.loads(mp.read_text(encoding='utf-8')); train=np.asarray(manifest['selected_indices'],dtype=np.int64)
            set_seed(seed); model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=0.2).to(device); init=initialize_f2(model,INIT)
            ds=CachedDataset(Path(role['fit']['path']),train); val_ds=CachedDataset(Path(role['fit']['path']),val); dl=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); criterion=nn.BCEWithLogitsLoss(); history=[]; best=-1.; best_epoch=0; outdir=OUT/condition; outdir.mkdir(parents=True,exist_ok=True); ck=outdir/f'{condition}__seed_{seed}.pth'; started=time.time()
            try:
                for epoch in range(1,9):
                    model.train(); losses=[]
                    for batch in dl:
                        x=batch['signal'].to(device,non_blocking=True); y=batch['label'].to(device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
                    yt,yl=predict_cached(model,val_ds,device,64); vm=metrics(yt,yl); history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':vm})
                    if vm['macro_aupr']>best: best=vm['macro_aupr']; best_epoch=epoch; torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'validation':vm},ck)
            finally: del ds
            model.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)['model_state_dict'])
            result={'schema':'route-a-if-sweep-c1-bce-v1','status':'completed','condition':condition,'seed':seed,'manifest':manifest,'backbone':'F2_DFMS_Full','interface':'complex_1d_true [B,2,3000]','loss':'BCEWithLogitsLoss','optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,'batch_size':32,'epochs':8,'initialization':init,'best_epoch':best_epoch,'validation':history[best_epoch-1]['validation'],'history':history,'test_metrics_by_power':{},'test_sampling':('full_role_indices' if args.full_test else f'deterministic_{args.test_samples}_per_power'),'cost_profile':profile_cost(model,device),'total_parameters':int(sum(p.numel() for p in model.parameters())),'elapsed_seconds':time.time()-started,'checkpoint_sha256':sha(ck),'locked_test_used':False,'test_files_opened':False,'frozen_e5_modified':False}
            if not args.skip_test:
                for power in POWERS_DB:
                    test_ds=CachedDataset(test_paths[int(power)],test_indices[int(power)]); yt,yl=predict_cached(model,test_ds,device,64); result['test_metrics_by_power'][str(power)]=metrics(yt,yl); del test_ds
                result['test_files_opened']=True
            (outdir/f'{condition}__seed_{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); all_rows.append(result); print(json.dumps({'condition':condition,'seed':seed,'val_macro_aupr':result['validation']['macro_aupr'],'test30_combo':None if args.skip_test else result['test_metrics_by_power']['30']['combination_exact_accuracy']},ensure_ascii=False),flush=True)
    summary={'schema':'route-a-if-sweep-c1-summary-v1','status':'completed','conditions':args.conditions,'seeds':args.seeds,'runs':len(all_rows),'locked_test_used':False,'frozen_e5_modified':False}
    for c in args.conditions:
        rr=[r for r in all_rows if r['condition']==c]; summary[c]={'n':len(rr),'effective_if_mean':float(np.mean([r['manifest']['effective_if'] for r in rr])),'effective_cf_mean':float(np.mean([r['manifest']['effective_cf'] for r in rr])),'validation_macro_aupr_mean':float(np.mean([r['validation']['macro_aupr'] for r in rr])),'validation_macro_f1_mean':float(np.mean([r['validation']['macro_f1'] for r in rr])),'validation_combo_mean':float(np.mean([r['validation']['combination_exact_accuracy'] for r in rr])),'test_by_power':{str(p):{'macro_aupr_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['macro_aupr'] for r in rr])),'macro_f1_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['macro_f1'] for r in rr])),'combo_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['combination_exact_accuracy'] for r in rr])),'hamming_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['hamming_loss'] for r in rr]))} for p in POWERS_DB}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'terminal_receipt.json').write_text(json.dumps({'status':'completed','summary':str((OUT/'summary.json').resolve()),'locked_test_used':False,'frozen_e5_modified':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','summary':str((OUT/'summary.json').resolve()),'runs':len(all_rows)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
