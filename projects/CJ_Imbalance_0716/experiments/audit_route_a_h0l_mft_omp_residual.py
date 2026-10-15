"""Strict provenance audit for H0l paired MFT OMP data."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]; DATA=Path(r'C:\LJQ_code\干扰数据'); COUNT=640
def source(name):return DATA/'h0l_mft_omp_residual_pairs_20260803'/name
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def receiver(p):
 with h5py.File(p,'r') as h:return np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64)
def run(a):
 seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'));receipt=json.loads(a.receipt.read_text(encoding='utf8'))
 if not receipt.get('executed') or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('receipt or test seal invalid')
 with h5py.File(a.low_path,'r') as lo,h5py.File(a.high_path,'r') as hi:
  lm={k:np.asarray(lo[k]) for k in lo.keys() if k not in ('iq_real','iq_imag')};hm={k:np.asarray(hi[k]) for k in hi.keys() if k not in ('iq_real','iq_imag')};shape_ok=lo['iq_real'].shape==hi['iq_real'].shape==(COUNT,3000);different=not np.array_equal(np.asarray(lo['iq_real'])[:,400:1000],np.asarray(hi['iq_real'])[:,400:1000])
 rid=lm['receiver_noise_seed'].reshape(-1).astype(np.int64);history=[]
 for old in sorted(DATA.rglob('*.mat')):
  if old.resolve() in {a.low_path.resolve(),a.high_path.resolve()} or 'test' in old.name.lower() or 'train' not in old.name.lower():continue
  try:history.append({'path':str(old),'receiver_overlap_count':int(np.intersect1d(rid,receiver(old)).size)})
  except (OSError,KeyError):continue
 same=('pair_seed','receiver_noise_seed','arrival_delay_samples','JNR_dB','delay_pool_samples','amplitude_pool')
 checks={'count_640':len(rid)==COUNT,'pair_identity_exact':np.array_equal(lm['pair_seed'],hm['pair_seed']) and len(np.unique(rid))==COUNT,'receiver_range_exact':(int(rid.min()),int(rid.max()))==(1058677744,1058678383),'non_count_metadata_exact':all(np.array_equal(lm[k],hm[k]) for k in same),'count_intervention_exact':np.all(lm['mft_num_targets']==3) and np.all(hm['mft_num_targets']==8),'payload_shape_exact':shape_ok,'iq_differs_in_declared_roi':different,'historical_receiver_overlap_zero':not any(x['receiver_overlap_count'] for x in history),'test_v3_sealed':True};checks={k:bool(v) for k,v in checks.items()}
 if not all(checks.values()):raise ValueError(f'H0l contract failed: {checks}')
 result={'status':'completed_train_only_paired_audit','experiment':'RouteA_H0l_MFTOMPResidualIdentifiability','low_sha256':sha(a.low_path),'high_sha256':sha(a.high_path),'sample_count_per_arm':COUNT,'receiver_noise_seed_range':[int(rid.min()),int(rid.max())],'contracts':checks,'live_historical_train_identity_audit':history,'identity_audit_passed':True,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0l_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2));return result
def args():
 p=argparse.ArgumentParser();p.add_argument('--low-path',type=Path,default=source('mft_count_3.mat'));p.add_argument('--high-path',type=Path,default=source('mft_count_8.mat'));p.add_argument('--receipt',type=Path,default=ROOT/'outputs/tapc/route_a_h0l_mft_omp_residual_generation/formal_generation_receipt.json');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0l_mft_omp_residual_generation/paired_identity_audit.json');return p.parse_args()
if __name__=='__main__':run(args())
