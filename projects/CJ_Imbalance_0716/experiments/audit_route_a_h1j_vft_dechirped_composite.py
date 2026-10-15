"""Audit new H1j composite source and fixed receiver-identity roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r'C:\LJQ_code\干扰数据')
COUNT, RECEIVER_RANGE = 5040, (998677743, 998682782)


def _source() -> Path:
    matches = sorted(DATA_ROOT.glob('*0802_h1j_vft_dechirped_composite'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one H1j source directory, found {matches}')
    return matches[0] / 'closedset_6class_train.mat'


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _receiver(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as handle:
        return np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    receipt = json.loads(args.generation_receipt.read_text(encoding='utf8'))
    if not receipt.get('executed') or any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('H1j source or sealed-test contract failed')
    if 'test' in args.data_path.name.lower():
        raise ValueError('H1j refuses a test-named source')
    with h5py.File(args.data_path, 'r') as handle:
        receiver = np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)
        labels = np.asarray(handle['label_selected']).T.astype(np.uint8)
    roles = {'fit': np.flatnonzero(receiver % 10 <= 5), 'validation': np.flatnonzero((receiver % 10 >= 6) & (receiver % 10 <= 7)), 'evaluation': np.flatnonzero(receiver % 10 >= 8)}
    history = []
    for old in sorted(DATA_ROOT.rglob('*.mat')):
        if old.resolve() == args.data_path.resolve() or 'test' in old.name.lower() or 'train' not in old.name.lower():
            continue
        try:
            history.append({'path': str(old), 'receiver_overlap_count': int(np.intersect1d(receiver, _receiver(old)).size)})
        except (OSError, KeyError):
            continue
    if any(row['receiver_overlap_count'] for row in history):
        raise ValueError('H1j historical receiver overlap')
    observed = (int(receiver.min()), int(receiver.max()))
    contracts = {
        'sample_count_5040': len(receiver) == COUNT,
        'receiver_unique_5040': len(np.unique(receiver)) == COUNT,
        'receiver_range_exact': observed == RECEIVER_RANGE,
        'six_labels_balanced_2560': labels.shape == (COUNT, 6) and bool(np.all(labels.sum(axis=0) == 2560)),
        'roles_cover_once': sum(len(value) for value in roles.values()) == COUNT and not any(np.intersect1d(roles[a], roles[b]).size for a, b in (('fit','validation'),('fit','evaluation'),('validation','evaluation'))),
        'role_counts_3024_1008_1008': [len(roles[name]) for name in ('fit','validation','evaluation')] == [3024,1008,1008],
        'test_v3_sealed': True,
    }
    if not all(contracts.values()):
        raise ValueError(f'H1j source contract failed: {contracts}')
    result = {
        'status': 'completed_train_only_identity_audit', 'experiment': 'RouteA_H1j_VFTDechirpedCompositeSource',
        'evidence_role': 'fresh_composite_vft_physical_source_validation', 'path': str(args.data_path), 'sha256': _sha(args.data_path),
        'sample_count': COUNT, 'receiver_noise_seed_range': list(observed), 'receiver_noise_seed_unique': int(len(np.unique(receiver))),
        'roles': {name: {'count': int(len(index)), 'receiver_count': int(len(np.unique(receiver[index]))), 'label_positive_counts': labels[index].sum(axis=0).astype(int).tolist()} for name, index in roles.items()},
        'live_historical_train_identity_audit': history, 'contracts': contracts, 'identity_audit_passed': True,
        'test_files_opened': False, 'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False, 'test_h1j_used': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    np.savez_compressed(args.roles_output, **roles)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=_source())
    parser.add_argument('--generation-receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/formal_generation_receipt.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/identity_audit.json')
    parser.add_argument('--roles-output', type=Path, default=ROOT / 'outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/role_indices.npz')
    parser.add_argument('--test-v3-manifest', type=Path, default=ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
