"""Audit the H0j paired MFT count intervention without opening any test data."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py, numpy as np

ROOT=Path(__file__).resolve().parents[1]; DATA_ROOT=Path(r'C:\LJQ_code\干扰数据'); COUNT=640
def _source(name):
    path=DATA_ROOT/'h0j_mft_matched_filter_support_pairs_20260802'/name
    if not path.is_file(): raise FileNotFoundError(path)
    return path
def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def _receiver(path):
    with h5py.File(path,'r') as h: return np.asarray(h['receiver_noise_seed']).reshape(-1).astype(np.int64)
def run(args):
    seal=json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if any(seal['test_v3'][key] for key in ('access_authorized','analysis_allowed','used')): raise ValueError('test-v3 is not sealed')
    receipt=json.loads(args.generation_receipt.read_text(encoding='utf8'))
    if not receipt.get('executed'): raise ValueError('formal H0j receipt missing')
    with h5py.File(args.low_path,'r') as low,h5py.File(args.high_path,'r') as high:
        low_meta={k:np.asarray(low[k]) for k in low.keys() if k not in ('iq_real','iq_imag')}; high_meta={k:np.asarray(high[k]) for k in high.keys() if k not in ('iq_real','iq_imag')}
        # MATLAB HDF5 exposes this row-major payload as [sample, time] in h5py.
        # Compare inside the declared MFT physical ROI, not the leading shared-noise region.
        iq_different=not np.array_equal(np.asarray(low['iq_real'])[:,400:1000],np.asarray(high['iq_real'])[:,400:1000])
        shapes_equal=low['iq_real'].shape==high['iq_real'].shape==(COUNT,3000) and low['iq_imag'].shape==high['iq_imag'].shape==(COUNT,3000)
    receiver=low_meta['receiver_noise_seed'].reshape(-1).astype(np.int64)
    equal_fields=('pair_seed','receiver_noise_seed','arrival_delay_samples','JNR_dB','delay_pool_samples','amplitude_pool')
    history=[]; excluded={args.low_path.resolve(),args.high_path.resolve()}
    for old in sorted(DATA_ROOT.rglob('*.mat')):
        if old.resolve() in excluded or 'test' in old.name.lower() or 'train' not in old.name.lower(): continue
        try: history.append({'path':str(old),'receiver_overlap_count':int(np.intersect1d(receiver,_receiver(old)).size)})
        except (OSError,KeyError): continue
    contracts={'count_exact':len(receiver)==COUNT,'pair_identity_exact':np.array_equal(low_meta['pair_seed'],high_meta['pair_seed']) and len(np.unique(receiver))==COUNT,
      'receiver_range_exact':bool(int(receiver.min())==1018677744 and int(receiver.max())==1018678383),
      'non_count_metadata_exact':all(np.array_equal(low_meta[k],high_meta[k]) for k in equal_fields),
      'count_intervention_exact':bool(np.all(low_meta['mft_num_targets']==3) and np.all(high_meta['mft_num_targets']==8)),
      'payload_shapes_exact':bool(shapes_equal),'iq_differs':bool(iq_different),'historical_receiver_overlap_zero':bool(not any(x['receiver_overlap_count'] for x in history)),'test_v3_sealed':True}
    if not all(contracts.values()): raise ValueError(f'H0j contract failed: {contracts}')
    result={'status':'completed_train_only_paired_audit','experiment':'RouteA_H0j_MFTMatchedFilterSupportIdentifiability','low_path':str(args.low_path),'high_path':str(args.high_path),'low_sha256':_sha(args.low_path),'high_sha256':_sha(args.high_path),'sample_count_per_arm':COUNT,'receiver_noise_seed_range':[int(receiver.min()),int(receiver.max())],'contracts':contracts,'live_historical_train_identity_audit':history,'identity_audit_passed':True,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0j_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); print(json.dumps(result,ensure_ascii=False,indent=2)); return result
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--low-path',type=Path,default=_source('mft_count_3.mat'));p.add_argument('--high-path',type=Path,default=_source('mft_count_8.mat'));p.add_argument('--generation-receipt',type=Path,default=ROOT/'outputs/tapc/route_a_h0j_mft_matched_filter_support_generation/formal_generation_receipt.json');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0j_mft_matched_filter_support_generation/paired_identity_audit.json');return p.parse_args()
if __name__=='__main__':run(parse_args())
