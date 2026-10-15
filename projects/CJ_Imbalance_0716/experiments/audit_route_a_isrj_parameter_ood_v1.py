"""Audit identity and ISRJ parameter contracts for external OOD v1."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import h5py, numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
 sys.path.insert(0,str(PROJECT_ROOT))
REFERENCES={
 "development_v3":(222677743,222702942),"test_v3_contract":(225677743,225702942),
 "development_v4":(232677743,232702942),"test_v4_contract":(235677743,235702942),
 "development_v5":(242677743,242702942),"test_v5_contract":(245677743,245702942),
 "development_v6":(252677743,252702942),"test_v6_contract":(255677743,255702942)}

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def run(args):
 with h5py.File(args.data_path,'r') as f:
  receiver=np.asarray(f['receiver_noise_seed']).reshape(-1).astype(np.int64)
  values=np.asarray(f['parameter_values']).T.astype(np.float64)
  mask=np.asarray(f['parameter_mask']).T.astype(bool)
 # Project schema is stable: ISRJ columns are resolved by the Python registry in evaluation;
 # audit expected active values by finite masked columns supplied explicitly.
 from cjlt.data.parameter_schema import get_parameter_specs
 width_col=get_parameter_specs('ISRJ')[1].column_index
 # Stable 79-column generator schema: ISRJ JNR/width/replicas occupy 52/53/54.
 replica_col=54
 width=values[mask[:,width_col],width_col]; replicas=values[mask[:,replica_col],replica_col]
 observed=(int(receiver.min()),int(receiver.max()))
 disjoint={name: bool(observed[1]<lo or observed[0]>hi) for name,(lo,hi) in REFERENCES.items()}
 payload={"status":"completed","evidence_role":"external_isrj_parameter_ood_v1",
  "path":str(args.data_path),"sha256":_sha(args.data_path),"sample_count":int(len(receiver)),
  "receiver_range":list(observed),"receiver_unique":int(np.unique(receiver).size),
  "identity_disjoint":disjoint,
  "isrj_sample_width_s_observed":[float(width.min()),float(width.max())],
  "isrj_num_replicas_observed":[int(replicas.min()),int(replicas.max())],
  "parameter_ood_passed":bool(width.min()>=22e-6-1e-9 and width.max()<=30e-6+1e-9 and replicas.min()>=6 and replicas.max()<=8),
  "identity_audit_passed":bool(len(receiver)==5040 and np.unique(receiver).size==5040 and all(disjoint.values())),
  "test_files_opened":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,
  "test_v4_used":False,"test_v5_used":False,"test_v6_used":False}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0801_isrj_ood_v1\closedset_6class_train.mat')); p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_isrj_parameter_ood_v1_generation/identity_parameter_audit.json'); run(p.parse_args())
