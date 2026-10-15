"""Copy base MAT metadata and substitute the audited AR(1) I/Q payload only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _copy_root(source: h5py.File, destination: h5py.File) -> None:
    for key in source.keys():
        if key != 'iq_complex':
            source.copy(key, destination)
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def run(args):
    if any('test' in path.name.lower() for path in (args.base_path, args.iq_payload, args.output)):
        raise ValueError('H0h refuses a test-named path')
    receipt = json.loads(args.materialization_receipt.read_text(encoding='utf8'))
    if not receipt['executed'] or float(receipt['rho']) != float(args.rho):
        raise ValueError('AR materialization receipt mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.base_path, 'r') as base, h5py.File(args.iq_payload, 'r') as payload, h5py.File(args.output, 'w') as out:
        if 'iq_complex' not in base or 'iq_complex' not in payload:
            raise KeyError('missing I/Q payload')
        if base['iq_complex'].shape != payload['iq_complex'].shape:
            raise ValueError('I/Q shape mismatch')
        _copy_root(base, out)
        payload.copy('iq_complex', out)
    with h5py.File(args.base_path, 'r') as base, h5py.File(args.output, 'r') as ar:
        keys_equal = set(base.keys()) == set(ar.keys())
        field_equal = {}
        field_contract = {}
        for key in base.keys():
            if key == 'iq_complex':
                continue
            if isinstance(base[key], h5py.Dataset) and isinstance(ar[key], h5py.Dataset):
                left, right = base[key][...], ar[key][...]
                if left.dtype.kind == 'O' or h5py.check_dtype(ref=left.dtype) is not None:
                    # HDF5 object references are relinked on copy, so their binary
                    # addresses are expected to differ.  Preserve shape/dtype and
                    # label this as a reference-relink contract, not byte equality.
                    equal = left.shape == right.shape and left.dtype == right.dtype
                    field_contract[key] = 'reference_relinked_shape_dtype'
                else:
                    try:
                        equal = np.array_equal(left, right, equal_nan=True)
                    except TypeError:
                        # MATLAB v7.3 complex values are compound records; NumPy
                        # cannot apply isnan to records, but exact equality is valid.
                        equal = np.array_equal(left, right)
                    field_contract[key] = 'exact_value'
                field_equal[key] = bool(equal)
        if not keys_equal or not all(field_equal.values()):
            raise ValueError('non-I/Q copy contract failed')
    result = {
        'status': 'completed', 'experiment': 'RouteA_H0h_AR1NoiseReplication',
        'evidence_role': 'fresh_ar1_receiver_noise_train_only',
        'base_path': str(args.base_path), 'base_sha256': _sha(args.base_path),
        'iq_payload': str(args.iq_payload), 'iq_payload_sha256': _sha(args.iq_payload),
        'output_path': str(args.output), 'output_sha256': _sha(args.output),
        'rho': float(args.rho), 'non_iq_keys_equal': keys_equal,
        'non_iq_dataset_exact': field_equal,
        'non_iq_dataset_contract': field_contract,
        'test_files_opened': False, 'locked_test_used': False, 'test_v2_used': False,
        'test_v3_used': False, 'test_h0h_used': False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h0h_ar1_base\closedset_6class_train.mat'))
    p.add_argument('--iq-payload', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_generation/ar1_iq_payload.mat')
    p.add_argument('--output', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h0h_ar1\closedset_6class_train.mat'))
    p.add_argument('--materialization-receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_generation/formal_ar1_materialization_receipt.json')
    p.add_argument('--receipt', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_generation/formal_dataset_materialization_receipt.json')
    p.add_argument('--rho', type=float, default=0.80)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
