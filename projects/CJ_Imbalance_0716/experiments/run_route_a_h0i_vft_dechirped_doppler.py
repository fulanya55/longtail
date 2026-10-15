"""No-fit H0i VFT physical screen; no classifier, threshold or calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_development_v2_h5 import _write_json
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals
from cjlt.models.vft_dechirped_doppler import dechirped_doppler_hz, fixed_phase_permutation, lfm_reference_phase


def _h0i_source(arm: str) -> Path:
    data_root = Path(r'C:\LJQ_code\干扰数据')
    matches = sorted(data_root.glob(f'*0802_h0i_vft_doppler_{arm}'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one H0i {arm} source directory, found {matches}')
    return matches[0] / 'closedset_6class_train.mat'


def _score(signals: np.ndarray, reference: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    output = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            output.append(dechirped_doppler_hz(batch, reference).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def _paired_bootstrap(low_c, high_c, low_k, high_k, count=2000, seed=20802026):
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(low_c)
    for _ in range(int(count)):
        index = rng.integers(0, n, size=n)
        label = np.r_[np.zeros(n), np.ones(n)]
        candidate = roc_auc_score(label, np.r_[low_c[index], high_c[index]])
        control = roc_auc_score(label, np.r_[low_k[index], high_k[index]])
        deltas.append(float(candidate - control))
    return [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]


def run(args):
    audit = json.loads(args.audit.read_text(encoding='utf8'))
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if not audit['identity_audit_passed'] or not all(audit['contracts'].values()):
        raise ValueError('H0i source audit failed')
    if any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('test-v3 is not sealed')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    low = _load_signals(args.low_path, np.arange(640), args.io_batch_size)
    high = _load_signals(args.high_path, np.arange(640), args.io_batch_size)
    reference = lfm_reference_phase(1000, device=device)
    control_reference = fixed_phase_permutation(reference)
    low_candidate, high_candidate = _score(low, reference, device, args.batch_size), _score(high, reference, device, args.batch_size)
    low_control, high_control = _score(low, control_reference, device, args.batch_size), _score(high, control_reference, device, args.batch_size)
    target = np.r_[np.zeros(len(low_candidate)), np.ones(len(high_candidate))]
    candidate_auroc = float(roc_auc_score(target, np.r_[low_candidate, high_candidate]))
    control_auroc = float(roc_auc_score(target, np.r_[low_control, high_control]))
    delta_ci = _paired_bootstrap(low_candidate, high_candidate, low_control, high_control)
    paired_rate = float(np.mean(high_candidate > low_candidate))
    checks = {
        'candidate_auroc_ge_0p65': candidate_auroc >= .65,
        'candidate_minus_control_auroc_ge_0p10': candidate_auroc - control_auroc >= .10,
        'candidate_minus_control_bootstrap_lower_gt_zero': delta_ci[0] > 0,
        'paired_high_gt_low_rate_ge_0p65': paired_rate >= .65,
        'no_fit_or_threshold': True,
        'test_v3_sealed': True,
    }
    result = {
        'status': 'completed', 'experiment': 'RouteA_H0i_VFTDechirpedDopplerIdentifiability',
        'evidence_role': 'fresh_no_fit_vft_physical_identifiability',
        'audit_path': str(args.audit), 'low_sha256': audit['low_sha256'], 'high_sha256': audit['high_sha256'],
        'statistic': {'pulse_samples': 1000, 'fs_Hz': 1e7, 'chirp_rate_Hz_per_s': 5e10, 'trained_parameters': 0,
                      'candidate': 'correct_known_LFM_dechirp_adjacent_phase_increment',
                      'control': 'fixed_permuted_LFM_phase_same_length_unit_modulus'},
        'candidate_auroc': candidate_auroc, 'control_auroc': control_auroc,
        'candidate_minus_control_auroc': candidate_auroc - control_auroc,
        'candidate_minus_control_paired_bootstrap_95_ci': delta_ci,
        'paired_high_greater_than_low_rate': paired_rate,
        'candidate_score_Hz': {'low_mean': float(low_candidate.mean()), 'high_mean': float(high_candidate.mean())},
        'control_score_Hz': {'low_mean': float(low_control.mean()), 'high_mean': float(high_control.mean())},
        'gates': checks, 'passed': bool(all(checks.values())),
        'external_refit': False, 'threshold_selected': False,
        'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False, 'test_h0i_used': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low-path', type=Path, default=_h0i_source('low'))
    parser.add_argument('--high-path', type=Path, default=_h0i_source('high'))
    parser.add_argument('--audit', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_h0i_vft_dechirped_doppler_generation/paired_identity_audit.json')
    parser.add_argument('--test-v3-manifest', type=Path, default=PROJECT_ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_h0i_vft_dechirped_doppler/route_a_h0i_vft_dechirped_doppler_summary.json')
    parser.add_argument('--io-batch-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
