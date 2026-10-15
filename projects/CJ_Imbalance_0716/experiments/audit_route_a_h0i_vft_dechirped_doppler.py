"""Train-only paired provenance audit for the no-fit H0i VFT screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r'C:\LJQ_code\干扰数据')
# In the 0716 six-class schema VFT fields are JNR, Doppler, velocity, delay.
# The generator makes velocity a deterministic physical derivative of Doppler.
COUNT, VFT_DOPPLER_COL, VFT_VELOCITY_COL = 640, 49, 50


def _h0i_source(arm: str) -> Path:
    matches = sorted(DATA_ROOT.glob(f'*0802_h0i_vft_doppler_{arm}'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one H0i {arm} source directory, found {matches}')
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


def _different_iq(low: h5py.File, high: h5py.File) -> bool:
    columns = np.asarray([0, 11, 37, 103, 211, 359, 509, 639])
    first = np.asarray(low['iq_complex'][:, columns])
    second = np.asarray(high['iq_complex'][:, columns])
    return bool(not np.array_equal(first, second))


def run(args):
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('test-v3 is not sealed')
    receipt = json.loads(args.generation_receipt.read_text(encoding='utf8'))
    if not receipt.get('executed'):
        raise ValueError('formal H0i source receipt missing')
    sources = (args.low_path, args.high_path)
    if any('test' in path.name.lower() for path in sources):
        raise ValueError('H0i rejects test-named data')
    with h5py.File(args.low_path, 'r') as low, h5py.File(args.high_path, 'r') as high:
        lr = np.asarray(low['receiver_noise_seed']).reshape(-1).astype(np.int64)
        hr = np.asarray(high['receiver_noise_seed']).reshape(-1).astype(np.int64)
        ll = np.asarray(low['label_selected']).T.astype(np.uint8)
        hl = np.asarray(high['label_selected']).T.astype(np.uint8)
        lv = np.asarray(low['parameter_values']).T.astype(np.float64)
        hv = np.asarray(high['parameter_values']).T.astype(np.float64)
        lm = np.asarray(low['parameter_mask']).T.astype(bool)
        hm = np.asarray(high['parameter_mask']).T.astype(bool)
        iq_differs = _different_iq(low, high)
    non_doppler = np.ones(lv.shape[1], dtype=bool)
    non_doppler[[VFT_DOPPLER_COL, VFT_VELOCITY_COL]] = False
    parameters_equal = np.array_equal(lv[:, non_doppler], hv[:, non_doppler], equal_nan=True)
    masks_equal = np.array_equal(lm, hm)
    low_doppler, high_doppler = lv[:, VFT_DOPPLER_COL], hv[:, VFT_DOPPLER_COL]
    history = []
    excluded = {args.low_path.resolve(), args.high_path.resolve()}
    for old in sorted(DATA_ROOT.rglob('*.mat')):
        if old.resolve() in excluded or 'test' in old.name.lower() or 'train' not in old.name.lower():
            continue
        try:
            overlap = int(np.intersect1d(lr, _receiver(old)).size)
            history.append({'path': str(old), 'receiver_overlap_count': overlap})
        except (OSError, KeyError):
            continue
    if any(row['receiver_overlap_count'] for row in history):
        raise ValueError('H0i historical receiver overlap')
    contracts = {
        'low_count_640': len(lr) == COUNT,
        'high_count_640': len(hr) == COUNT,
        'paired_receiver_identity_exact': np.array_equal(lr, hr) and len(np.unique(lr)) == COUNT,
        'vft_only_label_exact': bool(np.array_equal(ll, hl) and ll.shape == (COUNT, 1) and np.all(ll == 1)),
        'parameter_masks_exact': bool(masks_equal),
        'non_intervened_parameters_exact': bool(parameters_equal),
        'low_doppler_in_preregistered_band': bool(np.all((low_doppler >= 1e3) & (low_doppler <= 2.5e3))),
        'high_doppler_in_preregistered_band': bool(np.all((high_doppler >= 5.5e3) & (high_doppler <= 7e3))),
        'iq_differs_on_fixed_columns': iq_differs,
        'test_v3_sealed': True,
    }
    if not all(contracts.values()):
        raise ValueError(f'H0i contract failed: {contracts}')
    result = {
        'status': 'completed_train_only_paired_identity_audit',
        'experiment': 'RouteA_H0i_VFTDechirpedDopplerIdentifiability',
        'evidence_role': 'fresh_paired_vft_doppler_physical_screen',
        'low_path': str(args.low_path), 'low_sha256': _sha(args.low_path),
        'high_path': str(args.high_path), 'high_sha256': _sha(args.high_path),
        'sample_count_per_arm': COUNT, 'receiver_noise_seed_range': [int(lr.min()), int(lr.max())],
        'doppler_parameter_column': VFT_DOPPLER_COL, 'derived_velocity_parameter_column': VFT_VELOCITY_COL,
        'low_doppler_Hz_observed': [float(low_doppler.min()), float(low_doppler.max())],
        'high_doppler_Hz_observed': [float(high_doppler.min()), float(high_doppler.max())],
        'live_historical_train_identity_audit': history,
        'contracts': contracts, 'identity_audit_passed': True,
        'test_files_opened': False, 'locked_test_used': False, 'test_v2_used': False,
        'test_v3_used': False, 'test_h0i_used': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low-path', type=Path, default=_h0i_source('low'))
    parser.add_argument('--high-path', type=Path, default=_h0i_source('high'))
    parser.add_argument('--generation-receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0i_vft_dechirped_doppler_generation/formal_generation_receipt.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tapc/route_a_h0i_vft_dechirped_doppler_generation/paired_identity_audit.json')
    parser.add_argument('--test-v3-manifest', type=Path, default=ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
