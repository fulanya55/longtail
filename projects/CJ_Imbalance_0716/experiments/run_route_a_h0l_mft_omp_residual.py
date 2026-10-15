"""No-fit paired H0l OMP residual identifiability screen."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import h5py,numpy as np,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from cjlt.models.mft_omp_residual import fixed_phase_permutation,lfm_reference,three_atom_residual_fraction
from experiments.run_route_a_development_v2_h5 import _write_json
DATA=Path(r'C:\LJQ_code\干扰数据\h0l_mft_omp_residual_pairs_20260803')
def load(path):
 with h5py.File(path,'r') as h:z=np.asarray(h['iq_real'])+1j*np.asarray(h['iq_imag'])
 iq=np.stack((z.real,z.imag),axis=1).astype(np.float32);rms=np.sqrt(np.maximum(np.mean(iq[:,0]**2+iq[:,1]**2,axis=1,keepdims=True),1e-12));return iq/rms[:,:,None]
def score(x,ref,device,batch):
 out=[]
 with torch.no_grad():
  for s in range(0,len(x),batch):out.append(three_atom_residual_fraction(torch.from_numpy(x[s:s+batch]).to(device),ref).cpu().numpy())
 return np.concatenate(out).astype(np.float64)
def bootstrap(lo,hi,loctrl,hictrl):
 rng=np.random.default_rng(20802131);n=len(lo);y=np.r_[np.zeros(n),np.ones(n)];d=[]
 for _ in range(5000):
  ix=rng.integers(0,n,n);d.append(roc_auc_score(y,np.r_[lo[ix],hi[ix]])-roc_auc_score(y,np.r_[loctrl[ix],hictrl[ix]]))
 return [float(np.quantile(d,.025)),float(np.quantile(d,.975))]
def run(a):
 seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'));receipt=json.loads(a.receipt.read_text(encoding='utf8'))
 if not receipt.get('executed') or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('receipt or test seal invalid')
 device=torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'));lo,hi=load(a.low_path),load(a.high_path);ref=lfm_reference(device=device);ctrl=fixed_phase_permutation(ref);lc,hc=score(lo,ref,device,a.batch),score(hi,ref,device,a.batch);lk,hk=score(lo,ctrl,device,a.batch),score(hi,ctrl,device,a.batch);y=np.r_[np.zeros(len(lo)),np.ones(len(hi))];au=float(roc_auc_score(y,np.r_[lc,hc]));base=float(roc_auc_score(y,np.r_[lk,hk]));ci=bootstrap(lc,hc,lk,hk);paired=float(np.mean(hc>lc));checks={'candidate_auroc_ge_0p65':au>=.65,'candidate_minus_control_ge_0p10':au-base>=.10,'paired_bootstrap_lower_gt_zero':ci[0]>0,'paired_8_gt_3_rate_ge_0p65':paired>=.65,'zero_fit_threshold_calibration_fusion':True,'test_v3_sealed':True};result={'status':'completed','experiment':'RouteA_H0l_MFTOMPResidualIdentifiability','evidence_role':'fresh_no_fit_mft_sparse_attribution','statistic':{'candidate':'correct_LFM_fixed_three_atom_complex_OMP_residual_fraction','control':'fixed_phase_permuted_reference_same_dictionary','dictionary_delay_samples':[400,999],'atoms':3,'trained_parameters':0},'candidate_auroc':au,'control_auroc':base,'candidate_minus_control_auroc':au-base,'candidate_minus_control_paired_bootstrap_95_ci':ci,'paired_8_greater_than_3_rate':paired,'candidate_score':{'count3_mean':float(lc.mean()),'count8_mean':float(hc.mean())},'control_score':{'count3_mean':float(lk.mean()),'count8_mean':float(hk.mean())},'gates':checks,'passed':bool(all(checks.values())),'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h0l_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);_write_json(a.output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result
def args():
 p=argparse.ArgumentParser();p.add_argument('--low-path',type=Path,default=DATA/'mft_count_3.mat');p.add_argument('--high-path',type=Path,default=DATA/'mft_count_8.mat');p.add_argument('--receipt',type=Path,default=ROOT/'outputs/tapc/route_a_h0l_mft_omp_residual_generation/formal_generation_receipt.json');p.add_argument('--test-v3-manifest',type=Path,default=ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/tapc/route_a_h0l_mft_omp_residual/route_a_h0l_mft_omp_residual_summary.json');p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default=None);return p.parse_args()
if __name__=='__main__':run(args())
