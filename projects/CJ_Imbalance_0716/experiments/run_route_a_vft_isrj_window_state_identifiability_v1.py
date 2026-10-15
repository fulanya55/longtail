"""Frozen, no-fit phase-reset window-state screen on fresh paired data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import h5py
PROJECT_ROOT=Path(__file__).resolve().parents[1]
REGIMES=('W10_15_R3_4','W22_30_R3_4');VFT=4;ISRJ=5;WIDTH=53
def _score(iq,shuffle=False):
 out=[]
 for n,row in enumerate(np.asarray(iq,dtype=np.float32)):
  z=row[0].astype(float)+1j*row[1].astype(float)
  if shuffle: z=np.abs(z)*np.exp(1j*np.random.default_rng(20802000+n).permutation(np.angle(z)))
  a=np.abs(z); active=a>=.001*max(float(a.max()),1e-12); inc=np.angle(z[1:]*z[:-1].conj()); cur=np.abs(np.angle(np.exp(1j*(inc[1:]-inc[:-1])))); event=np.flatnonzero(active[:-2]&active[1:-1]&active[2:]&(cur>=.05))+1
  groups=np.split(event,np.flatnonzero(np.diff(event)>1)+1) if len(event) else []; centers=np.array([x.mean() for x in groups]); gaps=np.diff(centers); gaps=gaps[(gaps>=10)&(gaps<=500)]; out.append(float(np.median(gaps)) if len(gaps) else 0.)
 return np.asarray(out)
def _read(path):
 with h5py.File(path,'r') as h:
  y=np.asarray(h['label_selected'],dtype=np.uint8).T; p=np.asarray(h['parameter_values'],dtype=np.float64).T; m=(y[:,VFT]==0)&(y[:,ISRJ]==1); raw=np.asarray(h['iq_complex'][:,np.flatnonzero(m)]).T
 return y,p,m,np.stack([raw['real'],raw['imag']],axis=1)
def run(a):
 audit=json.loads(a.audit.read_text(encoding='utf8'));seal=json.loads(a.test_v3_manifest.read_text(encoding='utf8'))
 if not audit['passed'] or any(seal['test_v3'][k] for k in ('access_authorized','analysis_allowed','used')):raise ValueError('data or test-v3 boundary invalid')
 rows=[]
 for regime in REGIMES:
  p=a.data_root/regime/'closedset_6class_train.mat'; y,v,m,x=_read(p); rows.append((regime,_score(x),_score(x,True),v[m,WIDTH]))
 score=np.concatenate([r[1] for r in rows]);control=np.concatenate([r[2] for r in rows]);width=np.concatenate([r[3] for r in rows]);target=np.concatenate([np.full(len(r[1]),i,dtype=np.uint8) for i,r in enumerate(rows)])
 au=float(roc_auc_score(target,score));ac=float(roc_auc_score(target,control));rho=float(spearmanr(score,width).statistic)
 gates={'raw_phase_reset_auroc_ge_0p65':au>=.65,'raw_minus_scrambled_auroc_ge_0p05':au-ac>=.05,'raw_width_spearman_ge_0p30':rho>=.30,'test_v3_sealed':True}
 o={'status':'completed','experiment':'RouteA_VFTISRJ_WindowStateIdentifiabilityV1','evidence_role':'fresh_no_fit_window_state_screen','hard_negative_count_per_regime':{r[0]:len(r[1]) for r in rows},'raw_phase_reset_auroc':au,'phase_scrambled_auroc':ac,'auroc_delta':au-ac,'raw_width_spearman':rho,'gates':gates,'passed':bool(all(gates.values())),'external_refit':False,'hyperparameter_search':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(o,ensure_ascii=False,indent=2));return o
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_vft_isrj_windowstate_v1'));p.add_argument('--audit',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_vft_isrj_window_state_v1_generation/identity_parameter_pair_audit.json');p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_vft_isrj_window_state_identifiability_v1/summary.json');p.add_argument('--batch-size',type=int,default=128);run(p.parse_args())
if __name__=='__main__':main()
