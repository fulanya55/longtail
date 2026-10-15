"""Audit fresh v19 hard-negative source and write identity-isolated roles."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VFT, ISRJ, WIDTH, REPLICA = 4, 5, 53, 54
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def audit(data_path):
    with h5py.File(data_path,'r') as h:
        receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64); labels=np.asarray(h['label_selected'],dtype=np.uint8).T; values=np.asarray(h['parameter_values'],dtype=np.float64).T; mask=np.asarray(h['parameter_mask'],dtype=np.uint8).T.astype(bool)
    if len(receiver)!=5040 or np.unique(receiver).size!=5040: raise ValueError('v19 receiver identity contract failed')
    isrj=labels[:,ISRJ].astype(bool); eligible=isrj; width=values[isrj&mask[:,WIDTH],WIDTH]; replica=values[isrj&mask[:,REPLICA],REPLICA]
    if not(width.min()>=10e-6-1e-9 and width.max()<=30e-6+1e-9 and replica.min()>=3 and replica.max()<=4): raise ValueError('v19 ISRJ parameter contract failed')
    remainder=receiver%10; roles={'fit':np.flatnonzero(eligible&(remainder<=5)),'early_stop':np.flatnonzero(eligible&(remainder>=6)&(remainder<=7)),'evaluation':np.flatnonzero(eligible&(remainder>=8))}
    identities=[receiver[v] for v in roles.values()]
    if any(np.intersect1d(identities[i],identities[j]).size for i in range(3) for j in range(i+1,3)): raise ValueError('v19 receiver leakage')
    role_summary={}
    for name,idx in roles.items():
        positive=int(labels[idx,VFT].sum()); negative=int(len(idx)-positive)
        if positive<20 or negative<20: raise ValueError('v19 role lacks a target state')
        role_summary[name]={'count':int(len(idx)),'receiver_count':int(np.unique(receiver[idx]).size),'vft_positive_count':positive,'hard_negative_count':negative}
    payload={'status':'completed_train_only_identity_audit','evidence_role':'fresh_v19_hardnegative_vft_morphology_probe','path':str(data_path),'sha256':_sha(data_path),'sample_count':int(len(receiver)),'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'receiver_noise_seed_unique':int(np.unique(receiver).size),'isrj_positive_count':int(isrj.sum()),'width_range_s':[float(width.min()),float(width.max())],'replica_range':[int(replica.min()),int(replica.max())],'roles':role_summary,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v19_used':False}
    return payload,roles
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v19_vft_morphology_probe\closedset_6class_train.mat'));p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v19_vft_morphology_probe_generation/identity_audit.json');p.add_argument('--roles-output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v19_vft_morphology_probe_generation/role_indices.npz');a=p.parse_args();out,roles=audit(a.data_path);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');np.savez_compressed(a.roles_output,**roles);print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
