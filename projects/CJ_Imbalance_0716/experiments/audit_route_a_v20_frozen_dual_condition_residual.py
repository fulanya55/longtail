"""Audit v20 fresh identities and fixed fit/select/evaluation roles."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1];VFT,ISRJ,WIDTH,REPLICA=4,5,53,54
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def audit(path):
 with h5py.File(path,'r') as h: receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64);labels=np.asarray(h['label_selected'],dtype=np.uint8).T;values=np.asarray(h['parameter_values'],dtype=np.float64).T;mask=np.asarray(h['parameter_mask'],dtype=np.uint8).T.astype(bool)
 if len(receiver)!=5040 or np.unique(receiver).size!=5040:raise ValueError('v20 receiver contract failed')
 isrj=labels[:,ISRJ].astype(bool);w=values[isrj&mask[:,WIDTH],WIDTH];r=values[isrj&mask[:,REPLICA],REPLICA]
 if not(w.min()>=10e-6-1e-9 and w.max()<=30e-6+1e-9 and r.min()>=3 and r.max()<=4):raise ValueError('v20 parameter contract failed')
 rem=receiver%10;roles={'fit':np.flatnonzero(rem<=5),'selection':np.flatnonzero((rem>=6)&(rem<=7)),'evaluation':np.flatnonzero(rem>=8)};hard=(labels[:,VFT]==0)&isrj
 if any(np.intersect1d(receiver[roles[a]],receiver[roles[b]]).size for a in roles for b in roles if a<b):raise ValueError('v20 identity leakage')
 details={}
 for name,idx in roles.items():
  if hard[idx].sum()<20 or (labels[idx,VFT]==1).sum()<20:raise ValueError('v20 role lacks target cases')
  details[name]={'count':int(len(idx)),'receiver_count':int(np.unique(receiver[idx]).size),'hard_negative_count':int(hard[idx].sum()),'vft_positive_count':int((labels[idx,VFT]==1).sum()),'isrj_positive_count':int(isrj[idx].sum())}
 return {'status':'completed_train_only_identity_audit','evidence_role':'fresh_v20_frozen_dual_condition_residual','path':str(path),'sha256':_sha(path),'sample_count':int(len(receiver)),'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'receiver_noise_seed_unique':int(np.unique(receiver).size),'width_range_s':[float(w.min()),float(w.max())],'replica_range':[int(r.min()),int(r.max())],'roles':details,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v20_used':False},roles
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v20_dual_condition_residual\closedset_6class_train.mat'));p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v20_dual_condition_residual_generation/identity_audit.json');p.add_argument('--roles-output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v20_dual_condition_residual_generation/role_indices.npz');a=p.parse_args();out,roles=audit(a.data_path);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');np.savez_compressed(a.roles_output,**roles);print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
