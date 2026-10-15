"""Copy H0s metadata and substitute only the audited AR(1) I/Q payload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_root(source: h5py.File, destination: h5py.File) -> None:
    for key in source.keys():
        if key != 'iq_complex':
            source.copy(key, destination)
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def run(args):
    if any('test' in path.name.lower() for path in (args.base_path, args.iq_payload, args.output)):
        raise ValueError('H0s refuses a test-named path')
    receipt = json.loads(args.materialization_receipt.read_text(encoding='utf8'))
    if not receipt.get('executed', False) or float(receipt['rho']) != float(args.rho):
        raise ValueError('H0s AR materialization receipt mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.base_path, 'r') as base, h5py.File(args.iq_payload, 'r') as payload, h5py.File(args.output, 'w') as out:
        if 'iq_complex' not in base or 'iq_complex' not in payload:
            raise KeyError('missing I/Q payload')
        if base['iq_complex'].shape != payload['iq_complex'].shape:
            raise ValueError('I/Q shape mismatch')
        _copy_root(base, out)
        payload.copy('iq_complex', out)
    with h5py.File(args.base_path, 'r') as base, h5py.File(args.output, 'r') as derived:
        keys_equal = set(base.keys()) == set(derived.keys())
        exact, contracts = {}, {}
        for key in base.keys():
            if key == 'iq_complex':
                continue
            if not isinstance(base[key], h5py.Dataset) or not isinstance(derived[key], h5py.Dataset):
                # Groups were copied as complete HDF5 objects above.  The
                # source-compatible H0h materializer audits root datasets;
                # do the same instead of treating a group as an array.
                continue
            left, right = base[key][...], derived[key][...]
            if left.dtype.kind == 'O' or h5py.check_dtype(ref=left.dtype) is not None:
                equal = left.shape == right.shape and left.dtype == right.dtype
                contracts[key] = 'reference_relinked_shape_dtype'
            else:
                try:
                    equal = np.array_equal(left, right, equal_nan=True)
                except TypeError:
                    equal = np.array_equal(left, right)
                contracts[key] = 'exact_value'
            exact[key] = bool(equal)
        if not keys_equal or not all(exact.values()):
            raise ValueError('H0s non-I/Q copy contract failed')
    result = {
        'status': 'completed', 'experiment': args.experiment,
        'evidence_role': args.evidence_role,
        'base_path': str(args.base_path), 'base_sha256': _sha(args.base_path),
        'iq_payload': str(args.iq_payload), 'iq_payload_sha256': _sha(args.iq_payload),
        'output_path': str(args.output), 'output_sha256': _sha(args.output),
        'rho': float(args.rho), 'non_iq_keys_equal': keys_equal,
        'non_iq_dataset_exact': exact, 'non_iq_dataset_contract': contracts,
        'test_files_opened': False, 'locked_test_used': False, 'test_v2_used': False,
        'test_v3_used': False, f'test_{args.test_role_key}_used': False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1_base\closedset_6class_train.mat'))
    parser.add_argument('--iq-payload', type=Path, default=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_generation/ar1_iq_payload.mat')
    parser.add_argument('--output', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1\closedset_6class_train.mat'))
    parser.add_argument('--materialization-receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_generation/formal_ar1_materialization_receipt.json')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_generation/formal_dataset_materialization_receipt.json')
    parser.add_argument('--rho', type=float, default=0.80)
    parser.add_argument('--experiment', default='RouteA_H0s_ISRJCombinedParameterNoiseFrozen')
    parser.add_argument('--evidence-role', default='fresh_joint_parameter_receiver_noise_train_only')
    parser.add_argument('--test-role-key', default='h0s')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
