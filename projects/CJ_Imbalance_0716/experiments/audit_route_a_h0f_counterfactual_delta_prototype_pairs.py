"""Audit H0f fresh counterfactual-delta pair source without opening tests."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

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
    if any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')): raise ValueError('test-v3 sealed')
    with h5py.File(a.data_path,'r') as h:
        labels=np.asarray(h['label_selected'],dtype=np.uint8); receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64); pair=np.asarray(h['pair_id']).reshape(-1).astype(np.int64); target=np.asarray(h['pair_target_index']).reshape(-1).astype(np.int64)-1; side=np.asarray(h['pair_side']).reshape(-1).astype(np.int64); weak=np.asarray(h['weak_target']).reshape(-1).astype(np.int64); sample=np.asarray(h['sample_seed']).reshape(-1).astype(np.int64)
    if labels.shape!=(4800,6) or len(np.unique(pair))!=2400 or set(np.unique(target))!=set(TARGETS): raise ValueError('shape/target contract')
    roles={'fit':[],'early_stop':[],'evaluation':[]}; rows=[]
    for identifier in np.unique(pair):
        ix=np.flatnonzero(pair==identifier)
        if len(ix)!=2 or set(side[ix])!={0,1} or len(np.unique(target[ix]))!=1 or len(np.unique(receiver[ix]))!=1 or len(np.unique(sample[ix]))!=1: raise ValueError('pair metadata contract')
        neg,pos=ix[side[ix]==0][0],ix[side[ix]==1][0]; t=int(target[pos])
        if labels[neg,t]!=0 or labels[pos,t]!=1 or weak[neg]!=0 or weak[pos]!=1 or not np.array_equal(np.delete(labels[neg],t),np.delete(labels[pos],t)): raise ValueError('only target may change')
        role='fit' if receiver[pos]%10<=5 else 'early_stop' if receiver[pos]%10<=7 else 'evaluation'; roles[role].append(int(identifier)); rows.append((t,role))
    roles={k:np.asarray(v,dtype=np.int64) for k,v in roles.items()}
    if [len(roles[k]) for k in ('fit','early_stop','evaluation')]!=[1440,480,480]: raise ValueError('role count')
    counts={str(t):{k:sum(rt==t and rr==k for rt,rr in rows) for k in roles} for t in TARGETS}
    if any(v!=n for r in counts.values() for n,v in zip((480,160,160),(r['fit'],r['early_stop'],r['evaluation']))): raise ValueError('target balance')
    history=[]
    data_root=Path(r'C:\LJQ_code\干扰数据')
    for old in sorted(data_root.rglob('*.mat')):
        name=old.name.lower()
        if old.resolve()==a.data_path.resolve() or ('train' not in name and 'pair' not in name): continue
        try: history.append({'path':str(old),'receiver_overlap_count':int(np.intersect1d(np.unique(receiver),_receiver(old)).size)})
        except (OSError, KeyError): continue
    if any(r['receiver_overlap_count'] for r in history): raise ValueError('historical overlap')
    # H0e's generated MAT is no longer present under the data root.  Preserve
    # the recorded source range as a separately labelled metadata comparison;
    # do not misrepresent it as a live-file audit.
    h0e_audit=ROOT/'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json'
    recorded=[]
    if h0e_audit.exists():
        prior=json.loads(h0e_audit.read_text(encoding='utf8'))
        lo,hi=map(int,prior['receiver_noise_seed_range'])
        overlap=max(0,min(int(receiver.max()),hi)-max(int(receiver.min()),lo)+1)
        recorded.append({'artifact':str(h0e_audit),'source_path':prior.get('path'),'source_sha256':prior.get('sha256'),'receiver_noise_seed_range':[lo,hi],'receiver_range_overlap_count':overlap,'source_file_live':Path(prior.get('path','')).exists()})
    if any(r['receiver_range_overlap_count'] for r in recorded): raise ValueError('recorded historical range overlap')
    out={'status':'completed_train_only_pair_audit','experiment':'RouteA_H0f_CounterfactualDeltaPrototype','evidence_role':'fresh_train_only_counterfactual_delta_prototype_identifiability','path':str(a.data_path),'sha256':_sha(a.data_path),'pair_count':2400,'view_count':4800,'receiver_noise_seed_unique_pair_count':int(np.unique(receiver).size),'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'target_role_pair_counts':counts,'roles':{k:{'pair_count':int(len(v)),'view_count':int(2*len(v))} for k,v in roles.items()},'historical_train_identity_audit':history,'recorded_historical_source_audit':recorded,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0f_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); np.savez_compressed(a.roles_output,**roles); print(json.dumps(out,ensure_ascii=False,indent=2)); return out
def args():
    p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_counterfactual_delta_h0f\counterfactual_delta_pairs_train.mat')); p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0f_counterfactual_delta_generation/identity_audit.json'); p.add_argument('--roles-output',type=Path,default=ROOT/'outputs/tapc/route_a_h0f_counterfactual_delta_generation/role_pairs.npz'); p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json'); return p.parse_args()
if __name__=='__main__': run(args())
