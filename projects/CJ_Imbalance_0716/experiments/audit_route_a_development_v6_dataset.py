"""Audit train-only development-v6 without opening any test file."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
COUNT=25200
EXPECTED_RECEIVER=(252677743,252702942)
REFERENCES={
    "development_v2_train":(218938474,218963673),"test_v2_contract":(221938474,221963673),
    "development_v3_train":(222677743,222702942),"test_v3_contract":(225677743,225702942),
    "development_v4_train":(232677743,232702942),"test_v4_contract":(235677743,235702942),
    "development_v5_train":(242677743,242702942),"test_v5_contract":(245677743,245702942),
}
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def audit(path):
    path=Path(path).resolve()
    with h5py.File(path,'r') as f:
        receiver=np.asarray(f['receiver_noise_seed']).reshape(-1)
        labels=np.asarray(f['label_selected' if 'label_selected' in f else 'known_label'])
    observed=(int(receiver.min()),int(receiver.max()))
    if receiver.size!=COUNT or COUNT not in labels.shape or observed!=EXPECTED_RECEIVER or np.unique(receiver).size!=COUNT:
        raise ValueError('development-v6 identity contract failed')
    disjoint={name: observed[1]<value[0] or observed[0]>value[1] for name,value in REFERENCES.items()}
    if not all(disjoint.values()): raise ValueError('development-v6 overlaps prior role')
    return {"status":"completed","evidence_role":"development_v6","development_v6_path":str(path),
        "size_bytes":path.stat().st_size,"sha256":_sha(path),"sample_count":COUNT,
        "expected_sample_seed_range":[54000001,54025200],"observed_receiver_noise_seed_range":list(observed),
        "receiver_noise_seed_unique":COUNT,"reference_receiver_noise_seed_ranges":{k:list(v) for k,v in REFERENCES.items()},
        "receiver_identity_disjoint":disjoint,"identity_audit_passed":True,"test_files_opened":False,
        "locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"test_v4_used":False,
        "test_v5_used":False,"test_v6_used":False}
def run(args):
    payload=audit(args.data_path); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0801_v6\closedset_6class_train.mat'))
    p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs'/'tapc'/'route_a_development_v6_generation'/'identity_audit.json')
    return p.parse_args()
if __name__=='__main__': run(parse_args())
