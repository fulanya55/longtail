"""Build IF6 fixed-N manifests while preserving all 63 combinations."""
from __future__ import annotations
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import h5py,numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from cjlt.data.split_builder import combo_key
ROLE=ROOT/'outputs/route_a_c1_baseline_full_20260826/role_manifest.json'; OUT=ROOT/'outputs/route_a_if_sweep_20260902/manifests'; NAMES=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']
def sha(a): return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()
def main():
 role=json.loads(ROLE.read_text(encoding='utf-8')); source=Path(role['fit']['path']); fit=np.asarray(role['fit']['sample_indices'],dtype=np.int64); val=np.asarray(role['validation']['sample_indices'],dtype=np.int64)
 with h5py.File(str(source),'r') as h:y=np.asarray(h['known_label'][:,fit],dtype=np.uint8).T[:,[0,1,2,3,5,7]]
 groups=defaultdict(list)
 for i,row in enumerate(y):groups[combo_key(row)].append(i)
 names=sorted(groups); C=len(names); V=np.asarray([[int(b) for b in k] for k in names],float).T; avail=np.asarray([len(groups[k]) for k in names],float); target=8000*avail/avail.sum()
 # Variables: integer combo counts x, continuous positive/negative deviations.
 rows=[]; lo=[]; hi=[]
 def add_x(a,l,h): rows.append(np.concatenate([a,np.zeros(2*C)]));lo.append(l);hi.append(h)
 add_x(np.ones(C),8000,8000); h,t=0,5
 for i in range(6):
  if i!=t:add_x(V[i]-V[t],0,np.inf)
  if i!=h:add_x(V[h]-V[i],0,np.inf)
 add_x(V[h]-6*V[t],0,np.inf);add_x(V[h]-6.3*V[t],-np.inf,0);add_x(V[t],500,np.inf)
 for j in range(C):
  a=np.zeros(3*C);a[j]=1;a[C+j]=-1;a[2*C+j]=1;rows.append(a);lo.append(target[j]);hi.append(target[j])
 c=np.concatenate([np.zeros(C),1/np.maximum(target,1),1/np.maximum(target,1)])
 sol=milp(c,integrality=np.concatenate([np.ones(C),np.zeros(2*C)]),bounds=Bounds(np.concatenate([np.ones(C),np.zeros(2*C)]),np.concatenate([avail,np.full(2*C,np.inf)])),constraints=LinearConstraint(np.asarray(rows),np.asarray(lo),np.asarray(hi)),options={'time_limit':60.0})
 if not sol.success or sol.x is None: raise RuntimeError('IF6 combo-matched MILP infeasible: '+str(sol.message))
 x=np.rint(sol.x[:C]).astype(int); actual=(V@x).astype(int); records=[]
 for seed in (2039,2040,2041):
  rng=np.random.RandomState(seed); selected=[]
  for k,n in zip(names,x):
   cand=np.asarray(groups[k],dtype=np.int64).copy();rng.shuffle(cand);selected.extend(cand[:int(n)].tolist())
  local=np.asarray(selected,dtype=np.int64);rng.shuffle(local);global_idx=np.sort(fit[local]); yy=y[local];cc={k:int(n) for k,n in zip(names,x)}
  payload={'schema':'route-a-if6-combo-matched-manifest-v1','condition':'IF6_combo_matched','axis':'label_only_combo_preserved','seed':seed,'source':str(source.resolve()),'selected_indices':global_idx.tolist(),'selected_indices_sha256':sha(global_idx),'selected_count':len(global_idx),'requested_if':6.0,'achieved_label_counts':actual.tolist(),'effective_if':float(actual.max()/actual.min()),'combo_counts':cc,'effective_cf':float(x.max()/x.min()),'unique_combo_count':len(names),'validation_indices':val.tolist(),'validation_indices_sha256':sha(val),'locked_test_used':False,'frozen_e5_modified':False}
  (OUT/f'IF6_combo_matched__seed_{seed}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');records.append(payload)
 summary={'schema':'route-a-if6-combo-matched-summary-v1','status':'completed','selected_count':8000,'effective_if':float(actual.max()/actual.min()),'label_counts':actual.tolist(),'effective_cf':float(x.max()/x.min()),'unique_combo_count':len(names),'records':[{'seed':r['seed'],'sha256':r['selected_indices_sha256']} for r in records],'locked_test_used':False}
 (OUT.parent/'if6_combo_matched_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
