"""Evaluate existing IF-sweep checkpoints on the fixed C1 development tests."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import metrics
from experiments.run_route_a_if_sweep_c1_bce_20260902 import CachedDataset, predict_cached
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB

ROLE=ROOT/'outputs/route_a_c1_baseline_full_20260826/role_manifest.json'
RUNS=ROOT/'outputs/route_a_if_sweep_20260902/runs'
OUT=ROOT/'outputs/route_a_if_sweep_20260902/test_eval'

def predict_stream(model, path, indices, device, batch_size=64, chunk_size=512):
    """Evaluate in bounded chunks to avoid materializing large MAT views."""
    labels=[]; logits=[]
    idx=np.asarray(indices,dtype=np.int64)
    for start in range(0,len(idx),chunk_size):
        ds=CachedDataset(path,idx[start:start+chunk_size]); yt,yl=predict_cached(model,ds,device,batch_size); labels.append(yt); logits.append(yl); del ds
    return np.concatenate(labels), np.concatenate(logits)

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--conditions',nargs='+',default=['anchor_IF1','IF6','IF10','IF20']); ap.add_argument('--max-per-power',type=int,default=2048); ap.add_argument('--full-30',action='store_true'); args=ap.parse_args()
    device=torch.device('cuda:0'); role=json.loads(ROLE.read_text(encoding='utf-8')); OUT.mkdir(parents=True,exist_ok=True)
    paths={int(k):Path(v) for k,v in role['test']['paths_by_power'].items()}; indices={}
    for k,v in role['test']['sample_indices_by_power'].items():
        p=int(k); idx=np.asarray(v,dtype=np.int64)
        if not (args.full_30 and p==30) and len(idx)>args.max_per_power: idx=np.sort(np.random.default_rng(260826+p).choice(idx,size=args.max_per_power,replace=False))
        indices[p]=idx
    conditions={'anchor_IF1':[2039,2040,2041],'IF3':[2039,2040,2041],'IF5':[2039,2040,2041],'IF6':[2039,2040,2041,2042,2043],'IF10':[2039,2040,2041],'IF20':[2039,2040,2041],'IF6_combo_matched':[2039,2040,2041]}
    rows=[]
    conditions={c:conditions[c] for c in args.conditions}
    for c,seeds in conditions.items():
        for seed in seeds:
            ck=RUNS/c/f'{c}__seed_{seed}.pth'; payload=torch.load(str(ck),map_location='cpu',weights_only=False); model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=0.2).to(device); model.load_state_dict(payload.get('model_state_dict',payload),strict=True); result={'schema':'route-a-if-sweep-test-eval-v1','condition':c,'seed':seed,'test_metrics_by_power':{},'test_counts_by_power':{},'locked_test_used':False,'frozen_e5_modified':False}; started=time.time()
            for p in POWERS_DB:
                yt,yl=predict_stream(model,paths[int(p)],indices[int(p)],device,64,512); result['test_metrics_by_power'][str(p)]=metrics(yt,yl); result['test_counts_by_power'][str(p)]=int(len(indices[int(p)]))
            result['elapsed_seconds']=time.time()-started; (OUT/f'{c}__seed_{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); rows.append(result); print(json.dumps({'condition':c,'seed':seed,'test30_combo':result['test_metrics_by_power']['30']['combination_exact_accuracy']},ensure_ascii=False),flush=True)
    summary={'schema':'route-a-if-sweep-test-summary-v1','status':'completed','locked_test_used':False,'frozen_e5_modified':False,'conditions':conditions}
    for c,seeds in conditions.items():
        rr=[r for r in rows if r['condition']==c]; summary[c]={'n':len(rr),'test_by_power':{str(p):{'macro_aupr_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['macro_aupr'] for r in rr])),'macro_f1_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['macro_f1'] for r in rr])),'combo_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['combination_exact_accuracy'] for r in rr])),'hamming_mean':float(np.mean([r['test_metrics_by_power'][str(p)]['hamming_loss'] for r in rr]))} for p in POWERS_DB}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); (OUT/'terminal_receipt.json').write_text(json.dumps({'status':'completed','summary':str((OUT/'summary.json').resolve()),'locked_test_used':False},ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'status':'completed','summary':str((OUT/'summary.json').resolve()),'runs':len(rows)},ensure_ascii=False))

if __name__=='__main__': main()
