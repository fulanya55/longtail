"""Freeze three six-class composition-disjoint identity partitions for P5d."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from cj0724.dual_axis import combo_bits, read_identity_metadata

def digest(x): return hashlib.sha256(np.asarray(x,dtype='<i8').tobytes()).hexdigest()
def fold_for(combo: int) -> int:
    cardinality=int(combo).bit_count()
    return 0 if cardinality<=2 else 1 if cardinality==3 else 2
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-file',required=True);p.add_argument('--output-npz',required=True);p.add_argument('--output-json',required=True);p.add_argument('--seed',type=int,default=20260804);a=p.parse_args()
 ids,labels=read_identity_metadata(a.data_file); groups={}
 for i,row in enumerate(labels): groups.setdefault(int(combo_bits(row),2),[]).append(i)
 if len(groups)!=63 or any(len(v)!=100 for v in groups.values()): raise RuntimeError('P5d requires 63 combinations x 100 identities')
 rng=np.random.default_rng(a.seed); arrays={}; report={}
 for fold in range(3):
  role={k:[] for k in ('train','early','seen_confirm','unseen')}; unseen=[]
  for combo,values in sorted(groups.items()):
   order=rng.permutation(np.asarray(values,dtype=np.int64))
   if fold_for(combo)==fold: role['unseen'].append(order); unseen.append(combo)
   else:
    role['train'].append(order[:70]);role['early'].append(order[70:85]);role['seen_confirm'].append(order[85:100])
  for name,parts in role.items(): arrays[f'fold{fold}_{name}']=np.sort(np.concatenate(parts)).astype(np.int64)
  all_values=np.concatenate([np.concatenate(parts) for parts in role.values()]);
  if np.unique(all_values).size!=ids.size: raise RuntimeError('roles do not partition identities')
  report[str(fold)]={'unseen_combinations':unseen,'roles':{k:{'count':int(len(v)),'identity_sha256':digest(ids[v])} for k,v in ((n,arrays[f'fold{fold}_{n}']) for n in role)}}
 out=Path(a.output_npz);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays)
 payload={'seed':a.seed,'data_file':str(Path(a.data_file).resolve()),'identity_sha256':digest(ids),'folds':report,'role_rule':'unseen combination all evaluation-only; seen combination train/early/seen_confirm=70/15/15','test_generated':False,'test_opened':False,'locked_test_used':False};Path(a.output_json).write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
