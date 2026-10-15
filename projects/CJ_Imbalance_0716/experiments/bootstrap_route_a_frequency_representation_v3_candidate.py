"""Receiver-paired 5,000 bootstrap for the frozen V3 candidate."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'outputs/route_a_frequency_representation_v3/stage2_formal'; CAND=ROOT/'outputs/route_a_frequency_representation_v3/stage3_candidate'; SEEDS=(261021,261022,261023,261024,261025); CLS=6
def load(reg):
 out=[]
 for seed in SEEDS:
  s=json.loads((BASE/f'seed_{seed}'/reg/'summary.json').read_text()); best=max(('F0_B0_Focal','F1_B0_DB','F2_Ordinary_cRT','F3_RBC_cRT'),key=lambda a:s['results'][a]['by_view']['VH']['macro_aupr']); b=np.load(BASE/f'seed_{seed}'/reg/'evaluation_predictions.npz'); c=np.load(CAND/f'seed_{seed}'/reg/'evaluation_predictions.npz'); out.append((b['targets'],b[best+'_VH_logits'],c['VH_logits']))
 return out
def one(y,a,c,idx): return float(np.mean([average_precision_score(y[idx,j],c[idx,j])-average_precision_score(y[idx,j],a[idx,j]) for j in range(CLS)]))
def main():
 rng=np.random.RandomState(261099); rep=5000; result={}; t=time.time()
 for reg in ('LabelIF20','ComboIF20','FB'):
  cells=load(reg); samples=np.empty(rep,dtype=np.float64)
  for r in range(rep): samples[r]=np.mean([one(y,a,c,rng.randint(0,len(y),len(y))) for y,a,c in cells])
  result[reg]={'mean':float(samples.mean()),'95_ci':[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],'repetitions':rep,'positive_fraction':float(np.mean(samples>0))}
 # combined LabelIF20+ComboIF20 with paired cells retained
 cells=load('LabelIF20')+load('ComboIF20'); samples=np.empty(rep,dtype=np.float64)
 for r in range(rep): samples[r]=np.mean([one(y,a,c,rng.randint(0,len(y),len(y))) for y,a,c in cells])
 result['combined_primary']={'mean':float(samples.mean()),'95_ci':[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],'repetitions':rep,'positive_fraction':float(np.mean(samples>0))}
 out={'status':'completed','bootstrap_seed':261099,'results':result,'elapsed_seconds':time.time()-t,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False}; (CAND/'candidate_bootstrap_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
