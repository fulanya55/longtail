"""External paired transport of frozen v16 predicted ISRJ width."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import h5py, numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from experiments.audit_route_a_v17_isrj_width_transport import REGIMES, VFT, ISRJ, WIDTH, _read  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read as _read_json, _write  # noqa:E402
from experiments.run_route_a_development_v16_isrj_width_probe import WidthProbe, _load_signals  # noqa:E402

WIDE=REGIMES[1]

def _predict(model, signal, mean, std, device, batch=128):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(signal),batch):out.append(model(torch.from_numpy(signal[start:start+batch]).to(device)).cpu().numpy())
    return np.concatenate(out)*std+mean

def _row(seed, narrow, wide, normalizer, checkpoint_root, output_root, device):
    path=checkpoint_root/'runs'/'W_real'/f'seed_{seed}'/'best_model.pth'; item=torch.load(path,map_location=device,weights_only=False)
    model=WidthProbe(width=16).to(device);model.load_state_dict(item['state_dict'])
    nmask=(narrow['labels'][:,VFT]==0)&(narrow['labels'][:,ISRJ]==1); wmask=(wide['labels'][:,VFT]==0)&(wide['labels'][:,ISRJ]==1)
    if not np.array_equal(narrow['receiver'][nmask],wide['receiver'][wmask]):raise ValueError('hard-negative pairing changed')
    npred=_predict(model,narrow['signal'],normalizer['mean_seconds'],normalizer['std_seconds'],device)[nmask]
    wpred=_predict(model,wide['signal'],normalizer['mean_seconds'],normalizer['std_seconds'],device)[wmask]
    ntrue=narrow['values'][nmask,WIDTH];wtrue=wide['values'][wmask,WIDTH]
    score=np.concatenate([npred,wpred]); truth=np.concatenate([ntrue,wtrue]); regime=np.concatenate([np.zeros(len(npred)),np.ones(len(wpred))])
    row={'seed':int(seed),'hard_negative_count_per_regime':int(len(npred)),'predicted_width_delta_seconds':float((wpred-npred).mean()),'regime_auroc':float(roc_auc_score(regime,score)),'combined_r2':float(r2_score(truth,score)),'combined_spearman':float(spearmanr(truth,score).statistic),'parameter_count':int(sum(x.numel() for x in model.parameters()))}
    save=output_root/f'seed_{seed}';save.mkdir(parents=True,exist_ok=True);np.savez_compressed(save/'predictions.npz',receiver_noise_seed=narrow['receiver'][nmask],narrow_prediction=npred,wide_prediction=wpred,narrow_true=ntrue,wide_true=wtrue);_write(save/'metrics.json',row);return row

def run(args):
    audit=_read_json(args.audit); normalizer=_read_json(args.normalizer); v16=_read_json(args.v16_summary)
    if not audit.get('identity_audit_passed') or audit.get('test_files_opened') or not v16.get('gate',{}).get('passed'):raise ValueError('source audit or v16 gate invalid')
    if normalizer.get('source_data_sha256')!=v16['data']['sha256'] or not normalizer.get('fit_only'):raise ValueError('frozen normalizer contract invalid')
    narrow,wide=(_read(args.data_root/name/'closedset_6class_train.mat') for name in REGIMES)
    for entry in (narrow,wide):entry['signal']=_load_signals(args.data_root/(REGIMES[0] if entry is narrow else WIDE)/'closedset_6class_train.mat',np.arange(len(entry['receiver']),dtype=np.int64))
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu')); rows=[_row(seed,narrow,wide,normalizer,args.v16_root,args.output_root,device) for seed in args.seeds]
    aggregate={key:_mean_ci([row[key] for row in rows],2026080400+i) for i,key in enumerate(('predicted_width_delta_seconds','regime_auroc','combined_r2','combined_spearman'))}; aggregate['auroc_ge_0p75_seeds']=int(sum(row['regime_auroc']>=.75 for row in rows))
    checks={'matched_parameters_5_of_5':len({row['parameter_count'] for row in rows})==1,'predicted_shift_ge_3us':aggregate['predicted_width_delta_seconds']['mean']>=3e-6,'predicted_shift_ci_positive':aggregate['predicted_width_delta_seconds']['bootstrap_95_ci'][0]>0,'mean_regime_auroc_ge_0p75':aggregate['regime_auroc']['mean']>=.75,'regime_auroc_4_of_5':aggregate['auroc_ge_0p75_seeds']>=4,'mean_width_spearman_ge_0p40':aggregate['combined_spearman']['mean']>=.40}; passed=all(checks.values())
    payload={'status':'completed','experiment':'route_a_v17_isrj_predicted_width_transport','evidence_level':'fresh_external_physics_diagnostic','protocol':{'frozen_v16_checkpoints':str(args.v16_root),'seeds':[int(x) for x in args.seeds],'no_refit':True,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v17_used':False},'data_audit':audit,'per_seed':rows,'aggregate':aggregate,'gate':{'checks':checks,'passed':passed,'decision':'advance_to_frozen_q4_risk_alignment' if passed else 'stop_width_conditioned_correction'},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v17_used':False};args.output_root.mkdir(parents=True,exist_ok=True);_write(args.output_root/'route_a_v17_isrj_predicted_width_transport_summary.json',payload);_write(args.output_root/'run_receipt.json',{'status':'completed','gate_passed':passed,'next_decision':payload['gate']['decision'],'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_v17_used':False});print(json.dumps(payload,ensure_ascii=False,indent=2));return payload

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v17_isrj_width_transport'));p.add_argument('--audit',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v17_isrj_width_transport_generation/identity_parameter_pair_audit.json');p.add_argument('--v16-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v16_isrj_width_probe');p.add_argument('--v16-summary',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v16_isrj_width_probe/route_a_development_v16_isrj_width_probe_summary.json');p.add_argument('--normalizer',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v16_isrj_width_probe/width_normalizer.json');p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v17_isrj_predicted_width_transport');p.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041,2042,2043]);p.add_argument('--device',default=None);run(p.parse_args())
if __name__=='__main__':main()
