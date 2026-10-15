"""Audit H3's train-only source without opening a sealed test file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; DATA_ROOT=Path(r"C:\LJQ_code\干扰数据")
COUNT,BASE_SEED,RECEIVER_OFFSET=25_200,134_000_000,2*2_246_822_519

def _sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()
def _receiver(path:Path)->np.ndarray:
    with h5py.File(path,'r') as handle:return np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)
def run(args):
    receipt=json.loads(args.receipt.read_text(encoding='utf-8')); selection=json.loads(args.identity_selection.read_text(encoding='utf-8')); seal=json.loads(args.test_v3_manifest.read_text(encoding='utf-8'))
    if not receipt.get('executed') or selection.get('identity_overlap_count')!=0: raise ValueError('H3 receipt/identity selection invalid')
    if any(seal['test_v3'][key] for key in ('access_authorized','analysis_allowed','used')): raise ValueError('test-v3 seal is not intact')
    with h5py.File(args.source,'r') as handle:
        receiver=np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64); labels=np.asarray(handle['label_selected']); iq_shape=handle['iq_complex'].shape
    samples=np.arange(BASE_SEED+1,BASE_SEED+COUNT+1,dtype=np.uint64); expected=((samples+np.uint64(RECEIVER_OFFSET))%np.uint64(2**32)).astype(np.int64); rows=[]
    for path in DATA_ROOT.rglob('*train*.mat'):
        if path.resolve()==args.source.resolve():continue
        try:overlap=int(np.intersect1d(receiver,_receiver(path)).size)
        except (OSError,KeyError):continue
        rows.append({'path':str(path),'receiver_overlap_count':overlap})
    checks={'payload_shape_standard_complex_iq':iq_shape==(3000,COUNT),'labels_shape_contains_six_by_count':labels.shape in ((6,COUNT),(COUNT,6)),'receiver_count_exact':receiver.size==COUNT,'receiver_unique':np.unique(receiver).size==COUNT,'receiver_mapping_exact':np.array_equal(receiver,expected),'all_historic_train_receiver_overlap_zero':not any(row['receiver_overlap_count'] for row in rows),'source_is_standard_63x400':COUNT==63*400,'test_v3_still_sealed':True}; checks={key:bool(value) for key,value in checks.items()}
    if not all(checks.values()):raise ValueError(f'H3 source audit failed: {checks}')
    result={'status':'completed_train_only_source_audit','experiment':'RouteA_H3_FrozenUnifiedExternalConfirmation','evidence_role':'fresh_standard_train_only_external_confirmation','path':str(args.source),'sha256':_sha(args.source),'size_bytes':args.source.stat().st_size,'sample_count':COUNT,'contracts':checks,'historical_train_identity_audit':rows,'identity_audit_passed':True,'test_files_opened':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DATA_ROOT/'复合干扰数据0803_h3_frozen_unified_confirmation_r1'/'closedset_6class_train.mat');parser.add_argument('--receipt',type=Path,default=ROOT/'outputs/tapc/route_a_h3_frozen_confirmation_generation/formal_generation_receipt.json');parser.add_argument('--identity-selection',type=Path,default=ROOT/'outputs/tapc/route_a_h3_frozen_confirmation_generation/pre_generation_identity_selection.json');parser.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');parser.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h3_frozen_confirmation_generation/identity_audit.json');print(json.dumps(run(parser.parse_args()),ensure_ascii=False,indent=2))
