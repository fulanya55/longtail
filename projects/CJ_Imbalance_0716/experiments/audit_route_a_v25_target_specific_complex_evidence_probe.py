"""Audit fresh v25 source and write receiver-identity roles."""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import h5py,numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v25_target_complex_evidence\closedset_6class_train.mat'));p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v25_target_complex_evidence_generation/identity_audit.json');p.add_argument('--roles-output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v25_target_complex_evidence_generation/role_indices.npz');a=p.parse_args()
 with h5py.File(a.data_path,'r') as h: receiver=np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64);labels=np.asarray(h['label_selected'],dtype=np.uint8).T
 if len(receiver)!=5040 or np.unique(receiver).size!=5040:raise ValueError('v25 identity contract failed')
 rem=receiver%10;roles={'fit':np.flatnonzero(rem<=5),'early_stop':np.flatnonzero((rem>=6)&(rem<=7)),'evaluation':np.flatnonzero(rem>=8)}
 if any(np.intersect1d(receiver[roles[x]],receiver[roles[y]]).size for x,y in (('fit','early_stop'),('fit','evaluation'),('early_stop','evaluation'))):raise ValueError('v25 identity leakage')
 out={'status':'completed_train_only_identity_audit','evidence_role':'fresh_v25_target_specific_complex_evidence_probe','path':str(a.data_path),'sha256':_sha(a.data_path),'sample_count':int(len(receiver)),'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'receiver_noise_seed_unique':int(np.unique(receiver).size),'roles':{name:{'count':int(len(idx)),'receiver_count':int(np.unique(receiver[idx]).size),'label_positive_counts':[int(x) for x in labels[idx].sum(0)]} for name,idx in roles.items()},'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v25_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');np.savez_compressed(a.roles_output,**roles);print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
