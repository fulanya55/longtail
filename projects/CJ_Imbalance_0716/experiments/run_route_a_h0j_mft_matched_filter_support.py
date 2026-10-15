"""No-fit H0j MFT count identifiability screen with a valid physical control."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import h5py,numpy as np,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from cjlt.models.mft_matched_filter_support import effective_delay_support,fixed_phase_permutation,lfm_reference
from experiments.run_route_a_development_v2_h5 import _write_json
def _source(name):return Path(r'C:\LJQ_code\干扰数据\h0j_mft_matched_filter_support_pairs_20260802')/name
def _load(path):
 with h5py.File(path,'r') as h:
  # MATLAB HDF5 payload is exposed by h5py as [sample, time].
  z=np.asarray(h['iq_real'])+1j*np.asarray(h['iq_imag'])
 iq=np.stack((z.real,z.imag),axis=1).astype(np.float32)
 rms=np.sqrt(np.maximum(np.mean(iq[:,0]**2+iq[:,1]**2,axis=1,keepdims=True),1e-12));return iq/rms[:,:,None]
def _score(x,ref,device,batch):
 out=[]
 with torch.no_grad():
  for s in range(0,len(x),batch):out.append(effective_delay_support(torch.from_numpy(x[s:s+batch]).to(device),ref).cpu().numpy())
 return np.concatenate(out).astype(np.float64)
def _bootstrap(lo_c,hi_c,lo_k,hi_k,draws=5000,seed=20802112):
 rng=np.random.default_rng(seed); values=[];n=len(lo_c); y=np.r_[np.zeros(n),np.ones(n)]
 for _ in range(draws):
  i=rng.integers(0,n,n);values.append(roc_auc_score(y,np.r_[lo_c[i],hi_c[i]])-roc_auc_score(y,np.r_[lo_k[i],hi_k[i]]))
 return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]
def run(args):
 audit=json.loads(args.audit.read_text(encoding='utf8'));seal=json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
 if not audit['identity_audit_passed'] or not all(audit['contracts'].values()):raise ValueError('H0j audit failed')
 if any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('test-v3 is not sealed')
 device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'));lo,hi=_load(args.low_path),_load(args.high_path);ref=lfm_reference(device=device);control=fixed_phase_permutation(ref)
 lo_c,hi_c=_score(lo,ref,device,args.batch_size),_score(hi,ref,device,args.batch_size);lo_k,hi_k=_score(lo,control,device,args.batch_size),_score(hi,control,device,args.batch_size);y=np.r_[np.zeros(len(lo)),np.ones(len(hi))];candidate=float(roc_auc_score(y,np.r_[lo_c,hi_c]));baseline=float(roc_auc_score(y,np.r_[lo_k,hi_k]));ci=_bootstrap(lo_c,hi_c,lo_k,hi_k);paired=float(np.mean(hi_c>lo_c));checks={'candidate_auroc_ge_0p65':candidate>=.65,'candidate_minus_control_auroc_ge_0p10':candidate-baseline>=.10,'candidate_minus_control_paired_bootstrap_lower_gt_zero':ci[0]>0,'paired_8_gt_3_rate_ge_0p65':paired>=.65,'no_fit_or_threshold':True,'test_v3_sealed':True};result={'status':'completed','experiment':'RouteA_H0j_MFTMatchedFilterSupportIdentifiability','evidence_role':'fresh_no_fit_mft_count_physical_identifiability','statistic':{'candidate':'correct_LFM_matched_filter_Renyi2_effective_delay_support','control':'fixed_phase_permuted_unit_modulus_reference','roi_samples':[400,1000],'trained_parameters':0},'candidate_auroc':candidate,'control_auroc':baseline,'candidate_minus_control_auroc':candidate-baseline,'candidate_minus_control_paired_bootstrap_95_ci':ci,'paired_8_greater_than_3_rate':paired,'candidate_score':{'count3_mean':float(lo_c.mean()),'count8_mean':float(hi_c.mean())},'control_score':{'count3_mean':float(lo_k.mean()),'count8_mean':float(hi_k.mean())},'gates':checks,'passed':bool(all(checks.values())),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0j_used':False};args.output.parent.mkdir(parents=True,exist_ok=True);_write_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--low-path',type=Path,default=_source('mft_count_3.mat'));p.add_argument('--high-path',type=Path,default=_source('mft_count_8.mat'));p.add_argument('--audit',type=Path,default=ROOT/'outputs/tapc/route_a_h0j_mft_matched_filter_support_generation/paired_identity_audit.json');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0j_mft_matched_filter_support/route_a_h0j_mft_matched_filter_support_summary.json');p.add_argument('--batch-size',type=int,default=128);p.add_argument('--device',default=None);return p.parse_args()
if __name__=='__main__':run(parse_args())
