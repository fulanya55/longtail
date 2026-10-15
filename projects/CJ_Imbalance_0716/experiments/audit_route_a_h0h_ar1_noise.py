"""Train-only provenance audit for H0h AR(1) receiver-noise replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r'C:\LJQ_code\干扰数据')
COUNT = 5040
RECEIVER_RANGE = (968677743, 968682782)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as handle:
        return np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)


def _ranges_from_prior_audits() -> list[dict]:
    paths = {
        'H0e': ROOT / 'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json',
        'H0f': ROOT / 'outputs/tapc/route_a_h0f_counterfactual_delta_generation/identity_audit.json',
        'H0g': ROOT / 'outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_generation/identity_parameter_audit.json',
    }
    rows = []
    for name, path in paths.items():
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf8'))
            low, high = map(int, payload['receiver_noise_seed_range'])
            rows.append({'name': name, 'audit_artifact': str(path), 'range': [low, high]})
    return rows


def _finite_iq_difference(base: h5py.File, ar: h5py.File) -> bool:
    fixed_columns = np.array([0, 17, 71, 233, 509, 997, 1531, 2029, 2537, 3001, 3511, 4019, 4501, 5039])
    left = np.asarray(base['iq_complex'][:, fixed_columns])
    right = np.asarray(ar['iq_complex'][:, fixed_columns])
    return bool(np.isfinite(left['real']).all() and np.isfinite(left['imag']).all()
                and np.isfinite(right['real']).all() and np.isfinite(right['imag']).all()
                and not np.array_equal(left, right))


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('test-v3 is not sealed')
    if any('test' in path.name.lower() for path in (args.data_path, args.base_path)):
        raise ValueError('H0h refuses test-named data')
    material = json.loads(args.dataset_receipt.read_text(encoding='utf8'))
    process = json.loads(args.process_receipt.read_text(encoding='utf8'))
    if not material['status'] == 'completed' or not process.get('executed', False):
        raise ValueError('H0h materialization receipt is incomplete')
    if material['base_sha256'] != _sha(args.base_path) or material['output_sha256'] != _sha(args.data_path):
        raise ValueError('H0h materialization hash mismatch')
    if float(material['rho']) != 0.80 or float(process['rho']) != 0.80:
        raise ValueError('H0h rho mismatch')
    probe = process.get('process_probe', {})
    probe_ok = (
        int(probe.get('sample_count', 0)) == 200000
        and abs(float(probe.get('mean_power', np.nan)) - 1.0) <= float(probe.get('power_tolerance', 0.0))
        and float(probe.get('pseudocovariance_abs', np.inf)) <= 0.02
        and abs(float(probe.get('lag1_real_correlation', np.nan)) - 0.80) <= float(probe.get('correlation_tolerance', 0.0))
    )
    with h5py.File(args.base_path, 'r') as base, h5py.File(args.data_path, 'r') as ar:
        receiver = np.asarray(ar['receiver_noise_seed']).reshape(-1).astype(np.int64)
        labels = np.asarray(ar['label_selected']).T.astype(np.uint8)
        base_receiver = np.asarray(base['receiver_noise_seed']).reshape(-1).astype(np.int64)
        identity_exact = np.array_equal(receiver, base_receiver)
        labels_exact = np.array_equal(labels.T, np.asarray(base['label_selected']))
        iq_differs = _finite_iq_difference(base, ar)
    history = []
    excluded = {args.data_path.resolve(), args.base_path.resolve()}
    for old in sorted(DATA_ROOT.rglob('*.mat')):
        if old.resolve() in excluded or 'test' in old.name.lower() or 'train' not in old.name.lower():
            continue
        try:
            history.append({'path': str(old), 'receiver_overlap_count': int(np.intersect1d(receiver, _receiver(old)).size)})
        except (OSError, KeyError):
            continue
    if any(row['receiver_overlap_count'] for row in history):
        raise ValueError('live historical receiver overlap')
    recorded = _ranges_from_prior_audits()
    for row in recorded:
        low, high = row['range']
        row['receiver_range_overlap_count'] = max(0, min(int(receiver.max()), high) - max(int(receiver.min()), low) + 1)
    if any(row['receiver_range_overlap_count'] for row in recorded):
        raise ValueError('recorded receiver range overlap')
    observed = (int(receiver.min()), int(receiver.max()))
    contracts = {
        'sample_count_5040': len(receiver) == COUNT,
        'receiver_unique_5040': len(np.unique(receiver)) == COUNT,
        'receiver_range_exact': observed == RECEIVER_RANGE,
        'base_to_ar_identity_exact': bool(identity_exact),
        'base_to_ar_labels_exact': bool(labels_exact),
        'all_non_iq_dataset_contracts_passed': bool(all(material['non_iq_dataset_exact'].values())),
        'iq_changed_on_fixed_nonselection_columns': iq_differs,
        'rho_exactly_0p80': float(material['rho']) == 0.80,
        'proper_complex_ar1_process_probe_passed': bool(probe_ok),
        'test_v3_sealed': True,
    }
    if not all(contracts.values()):
        raise ValueError(f'H0h source contract failed: {contracts}')
    result = {
        'status': 'completed_train_only_identity_noise_audit',
        'experiment': 'RouteA_H0h_AR1NoiseReplication',
        'evidence_role': 'fresh_ar1_receiver_noise_frozen_replication',
        'path': str(args.data_path), 'sha256': _sha(args.data_path),
        'base_path': str(args.base_path), 'base_sha256': _sha(args.base_path),
        'sample_count': int(len(receiver)), 'receiver_noise_seed_range': list(observed),
        'receiver_noise_seed_unique': int(len(np.unique(receiver))),
        'noise_model': {'proper_complex_ar1_rho': 0.80, 'process_probe': probe},
        'dataset_materialization_receipt': str(args.dataset_receipt),
        'process_materialization_receipt': str(args.process_receipt),
        'live_historical_train_identity_audit': history,
        'recorded_counterfactual_range_audit': recorded,
        'contracts': contracts, 'identity_audit_passed': True,
        'test_files_opened': False, 'locked_test_used': False, 'test_v2_used': False,
        'test_v3_used': False, 'test_h0h_used': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h0h_ar1\closedset_6class_train.mat'))
    parser.add_argument('--base-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h0h_ar1_base\closedset_6class_train.mat'))
    parser.add_argument('--dataset-receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_generation/formal_dataset_materialization_receipt.json')
    parser.add_argument('--process-receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_generation/formal_ar1_materialization_receipt.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_generation/identity_noise_audit.json')
    parser.add_argument('--test-v3-manifest', type=Path, default=ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
