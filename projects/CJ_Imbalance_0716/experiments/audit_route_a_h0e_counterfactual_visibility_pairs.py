"""Identity and exact pair-contract audit for H0e CVMC source."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py, numpy as np

ROOT=Path(__file__).resolve().parents[1]; TARGETS=(3,4,5)
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def _receiver(path):
    with h5py.File(path,'r') as h: return np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64)
def run(a):
    seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'))
    if any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')): raise ValueError('test-v3 must remain sealed')
    with h5py.File(a.data_path,'r') as h:
        labels=np.asarray(h['label_selected'],dtype=np.uint8); receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64); pair=np.asarray(h['pair_id']).reshape(-1).astype(np.int64); target=np.asarray(h['pair_target_index']).reshape(-1).astype(np.int64)-1; side=np.asarray(h['pair_side']).reshape(-1).astype(np.int64); weak=np.asarray(h['weak_target']).reshape(-1).astype(np.int64); sample_seed=np.asarray(h['sample_seed']).reshape(-1).astype(np.int64)
    if labels.shape!=(4800,6) or len(np.unique(pair))!=2400 or not set(np.unique(target))==set(TARGETS): raise ValueError('H0e shape/target contract failed')
    pair_rows=[]; roles={'fit':[],'early_stop':[],'evaluation':[]}
    for identifier in np.unique(pair):
        ix=np.flatnonzero(pair==identifier)
        if len(ix)!=2 or set(side[ix])!={0,1} or len(np.unique(target[ix]))!=1 or len(np.unique(receiver[ix]))!=1 or len(np.unique(sample_seed[ix]))!=1: raise ValueError(f'pair {identifier} matching contract failed')
        neg,pos=ix[side[ix]==0][0],ix[side[ix]==1][0]; t=int(target[pos])
        if labels[neg,t]!=0 or labels[pos,t]!=1 or weak[neg]!=0 or weak[pos]!=1 or not np.array_equal(np.delete(labels[neg],t),np.delete(labels[pos],t)): raise ValueError(f'pair {identifier} label contract failed')
        role=('fit' if receiver[pos]%10<=5 else 'early_stop' if receiver[pos]%10<=7 else 'evaluation'); roles[role].append(int(identifier)); pair_rows.append({'pair_id':int(identifier),'target_index':t,'receiver_noise_seed':int(receiver[pos]),'role':role})
    roles={k:np.asarray(v,dtype=np.int64) for k,v in roles.items()}
    if [len(roles[x]) for x in ('fit','early_stop','evaluation')]!=[1440,480,480]: raise ValueError('role cardinality failed')
    per_target={str(t):{k:int(sum(row['target_index']==t and row['role']==k for row in pair_rows)) for k in roles} for t in TARGETS}
    if any(value!=count for row in per_target.values() for count,value in zip((480,160,160),(row['fit'],row['early_stop'],row['evaluation']))): raise ValueError('target role balance failed')
    history=[]
    for old in Path(r'C:\LJQ_code\干扰数据').glob('复合干扰数据*\\**\\closedset_6class_train.mat'):
        overlap=int(np.intersect1d(np.unique(receiver),_receiver(old)).size); history.append({'path':str(old),'receiver_overlap_count':overlap})
    if any(row['receiver_overlap_count'] for row in history): raise ValueError('historical identity overlap')
    out={'status':'completed_train_only_pair_audit','experiment':'RouteA_H0e_CounterfactualVisibilityMargin','evidence_role':'fresh_train_only_counterfactual_pair_identifiability','path':str(a.data_path),'sha256':_sha(a.data_path),'view_count':4800,'pair_count':2400,'receiver_noise_seed_unique_pair_count':int(np.unique(receiver).size),'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'target_role_pair_counts':per_target,'roles':{k:{'pair_count':int(len(v)),'view_count':int(2*len(v))} for k,v in roles.items()},'historical_train_identity_audit':history,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0e_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); np.savez_compressed(a.roles_output,**roles); print(json.dumps(out,ensure_ascii=False,indent=2)); return out
def args():
    p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_counterfactual_visibility_h0e\counterfactual_visibility_pairs_train.mat')); p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json'); p.add_argument('--roles-output',type=Path,default=ROOT/'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/role_pairs.npz'); p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json'); return p.parse_args()
if __name__=='__main__': run(args())
