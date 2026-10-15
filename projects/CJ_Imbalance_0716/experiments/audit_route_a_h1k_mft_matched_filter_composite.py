"""Train-only identity and fixed-role audit for the frozen H1k source."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; COUNT=5040; RANGE=(1028677743,1028682782)

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def receiver(path:Path)->np.ndarray:
    with h5py.File(path,'r') as h:return np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64)
def run(a):
    seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8')); receipt=json.loads(a.generation_receipt.read_text(encoding='utf8'))
    if not receipt.get('executed') or any(seal['test_v3'][x] for x in ('access_authorized','analysis_allowed','used')):raise ValueError('generation receipt or test seal invalid')
    if 'test' in a.data_path.name.lower():raise ValueError('H1k accepts train-only source only')
    rid=receiver(a.data_path)
    with h5py.File(a.data_path,'r') as h:labels=np.asarray(h['label_selected'],dtype=np.uint8).T
    roles={'fit':np.flatnonzero(rid%10<=5),'validation':np.flatnonzero((rid%10>=6)&(rid%10<=7)),'evaluation':np.flatnonzero(rid%10>=8)}
    history=[]; root=Path(r'C:\LJQ_code\干扰数据')
    for old in sorted(root.rglob('*.mat')):
        if old.resolve()==a.data_path.resolve() or 'test' in old.name.lower() or 'train' not in old.name.lower():continue
        try:history.append({'path':str(old),'receiver_overlap_count':int(np.intersect1d(rid,receiver(old)).size)})
        except (OSError,KeyError):continue
    if any(row['receiver_overlap_count'] for row in history):raise ValueError('historical receiver overlap')
    checks={'sample_count_5040':len(rid)==COUNT,'receiver_unique_5040':len(np.unique(rid))==COUNT,'receiver_range_exact':(int(rid.min()),int(rid.max()))==RANGE,'six_labels_balanced_2560':labels.shape==(COUNT,6) and bool(np.all(labels.sum(axis=0)==2560)),'all_63_combinations_present':len({tuple(x.tolist()) for x in labels})==63,'roles_cover_once':sum(len(x) for x in roles.values())==COUNT and not any(np.intersect1d(roles[x],roles[y]).size for x,y in (('fit','validation'),('fit','evaluation'),('validation','evaluation'))),'role_counts_3024_1008_1008':[len(roles[x]) for x in ('fit','validation','evaluation')]==[3024,1008,1008],'test_v3_sealed':True}
    if not all(checks.values()):raise ValueError(f'H1k source contract failed: {checks}')
    a.roles_output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.roles_output,**roles,receiver_noise_seed=rid)
    result={'status':'completed_train_only_identity_audit','experiment':'RouteA_H1k_MFTMatchedFilterComposite','evidence_role':'fresh_composite_mft_fixed_statistic_validation','path':str(a.data_path),'sha256':sha(a.data_path),'sample_count':COUNT,'receiver_noise_seed_range':[int(rid.min()),int(rid.max())],'receiver_noise_seed_unique':int(len(np.unique(rid))),'roles':{name:{'count':int(len(ids)),'receiver_count':int(len(np.unique(rid[ids]))),'label_positive_counts':labels[ids].sum(axis=0).astype(int).tolist()} for name,ids in roles.items()},'role_indices_path':str(a.roles_output),'role_indices_sha256':sha(a.roles_output),'live_historical_train_identity_audit':history,'contracts':checks,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h1k_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2));return result
def args():
    p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h1k_mft_support_composite\closedset_6class_train.mat'));p.add_argument('--generation-receipt',type=Path,default=ROOT/'outputs/tapc/route_a_h1k_mft_support_generation/formal_generation_receipt.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h1k_mft_support_generation/identity_audit.json');p.add_argument('--roles-output',type=Path,default=ROOT/'outputs/tapc/route_a_h1k_mft_support_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');return p.parse_args()
if __name__=='__main__':run(args())
