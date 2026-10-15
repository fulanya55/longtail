"""Fixed matched H1j source-level VFT validation; no calibration or threshold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from cjlt.models.vft_dechirped_doppler import dechirped_doppler_hz, fixed_phase_permutation, lfm_reference_phase
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals

VFT_INDEX = 4


def _source() -> Path:
    root = Path(r'C:\LJQ_code\干扰数据')
    matches = sorted(root.glob('*0802_h1j_vft_dechirped_composite'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one H1j directory, found {matches}')
    return matches[0] / 'closedset_6class_train.mat'


def _scores(signals: np.ndarray, reference: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows.append(dechirped_doppler_hz(batch, reference).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def _matched_view(labels: np.ndarray, weak: np.ndarray, *, evaluation: bool):
    other = [0, 1, 2, 3, 5]
    positive = labels[:, VFT_INDEX] == 1
    if evaluation:
        positive &= weak[:, VFT_INDEX]
    negative = labels[:, VFT_INDEX] == 0
    pos_context = {tuple(row) for row in labels[positive][:, other].tolist()}
    neg_context = {tuple(row) for row in labels[negative][:, other].tolist()}
    shared = pos_context & neg_context
    keep_pos = np.asarray([bool(positive[i]) and tuple(labels[i, other].tolist()) in shared for i in range(len(labels))])
    keep_neg = np.asarray([bool(negative[i]) and tuple(labels[i, other].tolist()) in shared for i in range(len(labels))])
    keep = keep_pos | keep_neg
    return keep, keep_pos[keep].astype(np.float32), {'positive_count': int(keep_pos.sum()), 'negative_count': int(keep_neg.sum()), 'matched_context_count': int(len(shared))}


def _fit_score(fit_score, fit_y, eval_score, seed, epochs, batch_size, device):
    mean, std = float(fit_score.mean()), max(float(fit_score.std()), 1e-6)
    train_x = torch.from_numpy(((fit_score - mean) / std).astype(np.float32))[:, None]
    train_y = torch.from_numpy(fit_y.astype(np.float32))[:, None]
    eval_x = torch.from_numpy(((eval_score - mean) / std).astype(np.float32))[:, None].to(device)
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if device.type == 'cuda': torch.cuda.manual_seed_all(int(seed))
    head = torch.nn.Linear(1, 1).to(device)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=0.)
    order_gen = torch.Generator().manual_seed(int(seed))
    for _ in range(int(epochs)):
        order = torch.randperm(len(train_x), generator=order_gen)
        for start in range(0, len(order), int(batch_size)):
            index = order[start:start + int(batch_size)]
            x, y = train_x[index].to(device), train_y[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(x), y)
            loss.backward(); optimizer.step()
    with torch.no_grad():
        probability = torch.sigmoid(head(eval_x)).squeeze(1).cpu().numpy().astype(np.float64)
    return probability, {'mean': mean, 'std': std, 'weight': float(head.weight.detach().cpu().item()), 'bias': float(head.bias.detach().cpu().item())}


def _rank(y, p):
    return {'auroc': float(roc_auc_score(y, p)), 'aupr': float(average_precision_score(y, p))}


def _run_seed(args, labels, weak, signals, roles, candidate, control, seed, device, data_sha):
    root = args.output_root / f'seed_{seed}'; receipt = root / 'seed_summary.json'
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding='utf8'))
    root.mkdir(parents=True, exist_ok=True)
    fit_ids, evaluation_ids = roles['fit'], roles['evaluation']
    fit_keep, fit_y, fit_counts = _matched_view(labels[fit_ids], weak[fit_ids], evaluation=False)
    eval_keep, eval_y, eval_counts = _matched_view(labels[evaluation_ids], weak[evaluation_ids], evaluation=True)
    strong_keep, strong_y, strong_counts = _matched_view(labels[evaluation_ids], ~weak[evaluation_ids], evaluation=True)
    # The strong view replaces weak membership by its complement without changing contexts.
    outputs = {}
    for name, score in (('CandidateCorrectDechirp', candidate), ('ControlPermutedPhase', control)):
        probability, head = _fit_score(score[fit_ids][fit_keep], fit_y, score[evaluation_ids][eval_keep], int(seed), args.epochs, args.batch_size, device)
        strong_probability = 1.0 / (1.0 + np.exp(-np.clip(((score[evaluation_ids][strong_keep] - head['mean']) / head['std']) * head['weight'] + head['bias'], -40, 40)))
        outputs[name] = {'weak': _rank(eval_y, probability), 'strong': _rank(strong_y, strong_probability), 'head': head}
    delta = {metric: outputs['CandidateCorrectDechirp']['weak'][metric] - outputs['ControlPermutedPhase']['weak'][metric] for metric in ('auroc','aupr')}
    delta['strong_auroc'] = outputs['CandidateCorrectDechirp']['strong']['auroc'] - outputs['ControlPermutedPhase']['strong']['auroc']
    payload = {'status': 'completed','seed': int(seed),'data_sha256': data_sha,'role_indices_sha256': _sha256(args.roles),
               'fit_matched_counts': fit_counts,'evaluation_weak_matched_counts': eval_counts,'evaluation_strong_matched_counts': strong_counts,
               'arms': outputs,'candidate_minus_control': delta,'trained_parameters_per_arm': 2,'epochs': int(args.epochs),
               'threshold_selected': False,'calibration_fit': False,'locked_test_used': False,'test_v2_used': False,'test_v3_used': False}
    _write_json(receipt,payload); return payload


def run(args):
    audit=json.loads(args.audit.read_text(encoding='utf8')); seal=json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if not audit['identity_audit_passed'] or not all(audit['contracts'].values()): raise ValueError('H1j audit failed')
    if any(seal['test_v3'][key] for key in ('access_authorized','analysis_allowed','used')): raise ValueError('test-v3 is not sealed')
    data_sha=_sha256(args.data_path)
    if data_sha.lower()!=audit['sha256'].lower(): raise ValueError('H1j data hash mismatch')
    labels=read_labels(args.data_path); params,_=read_parameters(args.data_path); weak=compute_weak_component_mask(labels,_component_power(params),args.weak_threshold_db)
    roles=np.load(args.roles); role_map={name:np.asarray(roles[name],dtype=np.int64) for name in ('fit','validation','evaluation')}
    signals=_load_signals(args.data_path,np.arange(len(labels)),args.io_batch_size); device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    reference=lfm_reference_phase(1000,device=device); candidate=_scores(signals,reference,device,args.score_batch_size); control=_scores(signals,fixed_phase_permutation(reference),device,args.score_batch_size)
    rows=[_run_seed(args,labels,weak,signals,role_map,candidate,control,seed,device,data_sha) for seed in args.seeds]
    aggregate={name:_mean_ci([row['candidate_minus_control'][name] for row in rows],20802100+i) for i,name in enumerate(('auroc','aupr','strong_auroc'))}
    checks={'weak_auroc_ge_0p015':aggregate['auroc']['mean']>=.015,'weak_auroc_lower_ci_gt_zero':aggregate['auroc']['bootstrap_95_ci'][0]>0,'weak_auroc_four_of_five':sum(row['candidate_minus_control']['auroc']>0 for row in rows)>=4,'weak_aupr_positive':aggregate['aupr']['mean']>0,'weak_aupr_four_of_five':sum(row['candidate_minus_control']['aupr']>0 for row in rows)>=4,'no_strong_auroc_loss_below_neg0p010':all(row['candidate_minus_control']['strong_auroc']>=-.010 for row in rows),'no_threshold_or_calibration':all(not row['threshold_selected'] and not row['calibration_fit'] for row in rows),'test_v3_sealed':True}
    result={'status':'completed','experiment':'RouteA_H1j_VFTDechirpedCompositeSource','evidence_role':'fresh_composite_vft_physical_source_validation','seeds':args.seeds,'data_sha256':data_sha,'statistic':'correct versus fixed phase-permuted known-LFM dechirped increment, each with matched one-dimensional zero-init BCE readout','aggregate_candidate_minus_control':aggregate,'per_seed':[{ 'seed':row['seed'],**row['candidate_minus_control']} for row in rows],'gate':{'checks':checks,'passed':bool(all(checks.values()))},'selected_or_promoted_model':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False,'test_h1j_used':False}
    args.output_root.mkdir(parents=True,exist_ok=True); _write_json(args.output_root/'route_a_h1j_vft_dechirped_composite_summary.json',result); print(json.dumps({'status':'completed','gate':result['gate'],'summary':str(args.output_root/'route_a_h1j_vft_dechirped_composite_summary.json')},ensure_ascii=False,indent=2));return result


def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--data-path',type=Path,default=_source());p.add_argument('--audit',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/identity_audit.json');p.add_argument('--roles',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/role_indices.npz');p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json');p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_h1j_vft_dechirped_composite');p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]);p.add_argument('--epochs',type=int,default=24);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--io-batch-size',type=int,default=128);p.add_argument('--score-batch-size',type=int,default=128);p.add_argument('--weak-threshold-db',type=float,default=10.0);p.add_argument('--device',default=None);p.add_argument('--resume',action='store_true');return p.parse_args()


if __name__=='__main__': run(parse_args())
