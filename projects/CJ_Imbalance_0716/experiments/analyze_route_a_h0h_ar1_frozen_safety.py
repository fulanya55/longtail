"""Non-selection safety reporting for the completed H0h frozen replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ('FMB', 'SawFM', 'SinFM', 'MFT', 'VFT', 'ISRJ')
VARIANTS = ('B0_H1_Frozen', 'B2_H1AlignedPeriodicity_Frozen')


def _fpr(target: np.ndarray, prediction: np.ndarray) -> float | None:
    negative = target == 0
    return None if not negative.any() else float(prediction[negative].mean())


def run(args):
    summary = json.loads(args.summary.read_text(encoding='utf8'))
    if summary['status'] != 'completed' or not summary['external_gate']['passed']:
        raise ValueError('H0h must be completed before non-selection safety reporting')
    with h5py.File(args.data_path, 'r') as handle:
        labels = np.asarray(handle['label_selected']).T.astype(np.uint8)
        receiver = np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(np.int64)
    per_seed = []
    for seed in summary['seeds']:
        seed_row = {'seed': int(seed), 'variants': {}}
        for variant in VARIANTS:
            path = args.output_root / f'seed_{seed}' / variant / 'predictions.npz'
            payload = np.load(path)
            if not np.array_equal(payload['targets'], labels):
                raise ValueError(f'target mismatch: {path}')
            prediction = payload['predictions'].astype(bool)
            by_label = {name: _fpr(labels[:, index], prediction[:, index]) for index, name in enumerate(CLASS_NAMES)}
            cohort = {}
            for value in range(5):
                mask = receiver % 5 == value
                cohort[str(value)] = {
                    'count': int(mask.sum()),
                    'fpr_by_label': {name: _fpr(labels[mask, index], prediction[mask, index]) for index, name in enumerate(CLASS_NAMES)},
                }
            seed_row['variants'][variant] = {'fpr_by_label': by_label, 'receiver_mod5_cohort': cohort}
        per_seed.append(seed_row)
    aggregate = {}
    for variant in VARIANTS:
        label_mean = {name: float(np.mean([row['variants'][variant]['fpr_by_label'][name] for row in per_seed])) for name in CLASS_NAMES}
        cohort_mean = {
            str(value): {
                name: float(np.mean([row['variants'][variant]['receiver_mod5_cohort'][str(value)]['fpr_by_label'][name] for row in per_seed]))
                for name in CLASS_NAMES
            }
            for value in range(5)
        }
        all_values = [value for value in label_mean.values()]
        all_cohort_values = [value for table in cohort_mean.values() for value in table.values()]
        aggregate[variant] = {
            'mean_fpr_by_label': label_mean,
            'mean_fpr_by_receiver_mod5_cohort_and_label': cohort_mean,
            'maximum_mean_label_fpr': float(max(all_values)),
            'maximum_mean_receiver_cohort_label_fpr': float(max(all_cohort_values)),
        }
    result = {
        'status': 'completed_nonselection_safety_report',
        'experiment': 'RouteA_H0h_AR1NoiseReplication_Frozen',
        'evidence_role': 'post_completed_nonselection_safety_description',
        'summary_path': str(args.summary), 'data_path': str(args.data_path),
        'receiver_cohort_definition': 'receiver_noise_seed modulo 5',
        'class_names': list(CLASS_NAMES), 'per_seed': per_seed, 'aggregate': aggregate,
        'selection_or_refit_performed': False,
        'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result['aggregate'], ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_h0h_ar1\closedset_6class_train.mat'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_frozen')
    parser.add_argument('--summary', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_frozen/route_a_h0h_ar1_noise_frozen_summary.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tapc/route_a_h0h_ar1_noise_frozen/nonselection_safety_report.json')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
