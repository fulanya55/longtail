"""Train-only identity and paired-nuisance audit for v17 width transport."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py, numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
ROOT=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v17_isrj_width_transport")
REGIMES=("W10_15_R3_4","W22_30_R3_4"); VFT,ISRJ,WIDTH,REPLICA=4,5,53,54

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def _read(path):
    with h5py.File(path,'r') as h:
        return {"receiver":np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64),"labels":np.asarray(h['label_selected'],dtype=np.uint8).T,"values":np.asarray(h['parameter_values'],dtype=np.float64).T,"mask":np.asarray(h['parameter_mask'],dtype=np.uint8).T.astype(bool),"legacy":np.asarray(h['data_legacy'],dtype=np.float32)}

def audit(root=ROOT):
    paths={name:Path(root)/name/'closedset_6class_train.mat' for name in REGIMES}; data={name:_read(path) for name,path in paths.items()}
    a,b=(data[name] for name in REGIMES); receiver=a['receiver']; isrj=a['labels'][:,ISRJ].astype(bool); absent=~isrj
    if len(receiver)!=1260 or np.unique(receiver).size!=1260 or (int(receiver.min()),int(receiver.max()))!=(728677743,728679002):raise ValueError('v17 receiver contract failed')
    if not np.array_equal(a['receiver'],b['receiver']) or not np.array_equal(a['labels'],b['labels']):raise ValueError('v17 regimes are not identity/label paired')
    if not np.array_equal(a['legacy'][:,absent],b['legacy'][:,absent]):raise ValueError('ISRJ-absent nuisance signals changed')
    wa=a['values'][isrj & a['mask'][:,WIDTH],WIDTH]; wb=b['values'][isrj & b['mask'][:,WIDTH],WIDTH]; ra=a['values'][isrj & a['mask'][:,REPLICA],REPLICA]; rb=b['values'][isrj & b['mask'][:,REPLICA],REPLICA]
    if not (wa.min()>=10e-6-1e-9 and wa.max()<=15e-6+1e-9 and wb.min()>=22e-6-1e-9 and wb.max()<=30e-6+1e-9 and np.array_equal(ra,rb) and ra.min()>=3 and ra.max()<=4):raise ValueError('v17 width/replica contract failed')
    return {"status":"completed_train_only_paired_audit","evidence_role":"fresh_external_predicted_width_transport","regimes":{name:{"path":str(paths[name]),"sha256":_sha(paths[name]),"sample_count":1260} for name in REGIMES},"receiver_noise_seed_range":[int(receiver.min()),int(receiver.max())],"receiver_noise_seed_unique":1260,"isrj_positive_count":int(isrj.sum()),"hard_negative_count":int(((a['labels'][:,VFT]==0)&isrj).sum()),"width_ranges_s":{REGIMES[0]:[float(wa.min()),float(wa.max())],REGIMES[1]:[float(wb.min()),float(wb.max())]},"replica_range":[int(ra.min()),int(ra.max())],"paired_labels":True,"isrj_absent_iq_exact":True,"identity_audit_passed":True,"test_files_opened":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"test_v17_used":False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v17_isrj_width_transport_generation/identity_parameter_pair_audit.json');a=p.parse_args();o=audit(a.data_root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(o,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
