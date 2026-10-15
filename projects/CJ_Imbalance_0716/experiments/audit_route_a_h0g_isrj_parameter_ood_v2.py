"""Train-only identity/parameter audit for the H0g frozen OOD replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNT = 5040
RECEIVER_RANGE = (958677743, 958682782)
ISRJ_WIDTH_COL = 53
ISRJ_REPLICA_COL = 54


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as handle:
        return np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)


def _recorded_ranges() -> list[dict]:
    paths = {
        'H0e': ROOT / 'outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json',
        'H0f': ROOT / 'outputs/tapc/route_a_h0f_counterfactual_delta_generation/identity_audit.json',
    }
    rows = []
    for name, path in paths.items():
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf8'))
            low, high = map(int, payload['receiver_noise_seed_range'])
            rows.append({'name': name, 'audit_artifact': str(path), 'range': [low, high]})
    return rows


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('test-v3 is not sealed')
    if 'test' in args.data_path.name.lower():
        raise ValueError('refusing a test-named source')
    with h5py.File(args.data_path, 'r') as handle:
        receiver = np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)
        values = np.asarray(handle['parameter_values']).T.astype(np.float64)
        masks = np.asarray(handle['parameter_mask']).T.astype(bool)
        labels = np.asarray(handle['label_selected']).T.astype(np.uint8)
    isrj = labels[:, 5].astype(bool)
    width_active = masks[:, ISRJ_WIDTH_COL] & isrj & np.isfinite(values[:, ISRJ_WIDTH_COL])
    replica_active = masks[:, ISRJ_REPLICA_COL] & isrj & np.isfinite(values[:, ISRJ_REPLICA_COL])
    widths = values[width_active, ISRJ_WIDTH_COL]
    replicas = values[replica_active, ISRJ_REPLICA_COL]
    if not len(widths) or not len(replicas):
        raise ValueError('missing active finite ISRJ parameters')
    history = []
    root = Path(r'C:\LJQ_code\干扰数据')
    for old in sorted(root.rglob('*.mat')):
        if old.resolve() == args.data_path.resolve() or 'test' in old.name.lower() or 'train' not in old.name.lower():
            continue
        try:
            history.append({'path': str(old), 'receiver_overlap_count': int(np.intersect1d(receiver, _receiver(old)).size)})
        except (OSError, KeyError):
            continue
    if any(row['receiver_overlap_count'] for row in history):
        raise ValueError('live historical receiver overlap')
    recorded = _recorded_ranges()
    for row in recorded:
        low, high = row['range']
        row['receiver_range_overlap_count'] = max(0, min(int(receiver.max()), high) - max(int(receiver.min()), low) + 1)
    if any(row['receiver_range_overlap_count'] for row in recorded):
        raise ValueError('recorded source range overlap')
    observed = (int(receiver.min()), int(receiver.max()))
    contract = {
        'sample_count_5040': len(receiver) == COUNT,
        'receiver_unique_5040': len(np.unique(receiver)) == COUNT,
        'receiver_range_exact': observed == RECEIVER_RANGE,
        'isrj_width_in_25_30us': float(widths.min()) >= 25e-6 - 1e-10 and float(widths.max()) <= 30e-6 + 1e-10,
        'isrj_replicas_exactly_9': int(replicas.min()) == 9 and int(replicas.max()) == 9,
        'test_v3_sealed': True,
    }
    if not all(contract.values()):
        raise ValueError(f'H0g source contract failed: {contract}')
    result = {
        'status': 'completed_train_only_identity_parameter_audit',
        'experiment': 'RouteA_H0g_ISRJParameterOODv2',
        'evidence_role': 'fresh_far_parameter_ood_frozen_replication',
        'path': str(args.data_path), 'sha256': _sha(args.data_path),
        'sample_count': int(len(receiver)), 'receiver_noise_seed_range': list(observed),
        'receiver_noise_seed_unique': int(len(np.unique(receiver))),
        'isrj_eligible_count': int(isrj.sum()),
        'isrj_sample_width_s_observed': [float(widths.min()), float(widths.max())],
        'isrj_num_replicas_observed': [int(replicas.min()), int(replicas.max())],
        'live_historical_train_identity_audit': history,
        'recorded_counterfactual_range_audit': recorded,
        'contracts': contract, 'identity_audit_passed': True,
        'test_files_opened': False, 'locked_test_used': False, 'test_v2_used': False,
        'test_v3_used': False, 'test_h0g_used': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_ood_h0g\closedset_6class_train.mat'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_generation/identity_parameter_audit.json')
    parser.add_argument('--test-v3-manifest', type=Path, default=ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
