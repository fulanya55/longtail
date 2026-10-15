"""Audit the third fresh source for frozen predicted-width H0c."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py, numpy as np

ROOT=Path(__file__).resolve().parents[1]; ISRJ,WIDTH,REPLICA=5,53,54
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_predicted_width_h0c\closedset_6class_train.mat')); p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0c_isrj_predicted_width_generation/identity_audit.json'); p.add_argument('--roles-output',type=Path,default=ROOT/'outputs/tapc/route_a_h0c_isrj_predicted_width_generation/role_indices.npz'); a=p.parse_args()
    with h5py.File(a.data_path,'r') as h: receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64); labels=np.asarray(h['label_selected'],dtype=np.uint8).T; values=np.asarray(h['parameter_values'],dtype=np.float64).T; mask=np.asarray(h['parameter_mask'],dtype=np.uint8).T.astype(bool)
    active=np.flatnonzero(labels[:,ISRJ]==1); width,replicas=values[active,WIDTH],values[active,REPLICA]
    if len(receiver)!=5040 or np.unique(receiver).size!=5040 or len(active)!=2560 or not np.all(mask[active,WIDTH]&mask[active,REPLICA]) or width.min()<10e-6-1e-9 or width.max()>30e-6+1e-9 or replicas.min()<3 or replicas.max()>8: raise ValueError('H0c source contract failed')
    rem=receiver%10; roles={'fit':active[rem[active]<=5],'early_stop':active[(rem[active]>=6)&(rem[active]<=7)],'evaluation':active[rem[active]>=8]}
    if [len(roles[x]) for x in ('fit','early_stop','evaluation')]!=[1536,512,512] or any(np.intersect1d(receiver[roles[x]],receiver[roles[y]]).size for x in roles for y in roles if x<y): raise ValueError('H0c role contract failed')
    history=[]
    for old in Path(r'C:\LJQ_code\干扰数据').glob('复合干扰数据*\**\closedset_6class_train.mat'):
        if '0802_isrj_predicted_width_h0c' in str(old): continue
        try:
            with h5py.File(old,'r') as h: old_receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64)
            history.append({'path':str(old),'receiver_overlap_count':int(np.intersect1d(receiver,old_receiver).size)})
        except OSError: continue
    if any(x['receiver_overlap_count'] for x in history): raise ValueError('historical overlap')
    out={'status':'completed_train_only_identity_audit','experiment':'RouteA_H0c_ISRJPredictedWidthSubstitution','evidence_role':'fresh_train_only_frozen_predicted_width_replica_identifiability','path':str(a.data_path),'sha256':_sha(a.data_path),'sample_count':int(len(receiver)),'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'receiver_noise_seed_unique':int(np.unique(receiver).size),'isrj_eligible_count':int(len(active)),'width_range_s':[float(width.min()),float(width.max())],'replica_range':[int(replicas.min()),int(replicas.max())],'roles':{k:{'count':int(len(v)),'receiver_count':int(np.unique(receiver[v]).size),'label_positive_counts':[int(x) for x in labels[v].sum(0)]} for k,v in roles.items()},'historical_train_identity_audit':history,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0c_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); np.savez_compressed(a.roles_output,**roles); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
