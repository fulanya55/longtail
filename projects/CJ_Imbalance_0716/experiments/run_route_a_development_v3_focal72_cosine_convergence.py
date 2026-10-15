"""One fixed 72-epoch cosine Focal convergence audit on development-v3."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import _component_power, _predict
from experiments.run_route_a_development_v3_mainstream_loss_baselines import (
    EXPECTED_V3_SHA256, METRICS, _comparison, _load_model_from_state, _load_rawsaw,
    _reference_result, _route_rawsaw, _train_loss_model, _value,
)
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _evaluate, _load_roles, _validate_config
from experiments.run_shared import _data_path, load_config

VARIANT = 'E5_Focal_72_Cosine'
FOCAL36 = 'Focal36_Reference'
H1 = 'H1_FrozenLinearBCE_Reference'
F3 = 'F3_AlignedPeriodicity_Reference'
F2_MODEL_NAME = 'F2_DFMS_Full'


def _variant_map(row):
    return {item['variant']: item for item in row['variants']}


def _aggregate(rows, names):
    return {
        name: {
            metric: _mean_ci([_value(_variant_map(row)[name], metric) for row in rows], 20802000 + ni * 100 + mi)
            for mi, metric in enumerate(METRICS)
        }
        for ni, name in enumerate(names)
    }


def _late_history_summary(path: Path) -> dict:
    lines = path.read_text(encoding='utf8').strip().splitlines()
    rows = [line.split(',') for line in lines[1:]]
    macro = [float(row[2]) for row in rows]
    weak = [float(row[3]) for row in rows]
    tail = min(10, len(rows))
    return {
        'epochs_recorded': len(rows),
        'last_ten_macro_aupr_change': float(macro[-1] - macro[-tail]),
        'last_ten_target_weak_auroc_change': float(weak[-1] - weak[-tail]),
        'final_classifier_lr': float(rows[-1][4]),
        'final_shared_lr': float(rows[-1][5]),
    }


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f'seed_{seed}'
    receipt = root / 'seed_summary.json'
    if args.resume and receipt.exists():
        return json.loads(receipt.read_text(encoding='utf8'))
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config['normalization'], args.io_batch_size)
    f2_dir = args.f2_root / 'runs' / F2_MODEL_NAME / f'seed_{seed}'
    template, f2_checkpoint = _load_backbone(f2_dir, F2_MODEL_NAME, device)
    initial_state = copy.deepcopy({key: value.detach().cpu() for key, value in template.state_dict().items()})
    del template
    raw, raw_checkpoint = _load_rawsaw(args, seed, device)
    from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _raw_scores
    raw_scores = {role: _raw_scores(raw, values['signals'], device, args.raw_batch_size) for role, values in roles.items()}
    del raw
    rows = [
        _reference_result(args.focal36_root, seed, 'E2_Focal_36', FOCAL36),
        _reference_result(args.head_root, seed, 'H1_Linear_BCE', H1),
        _reference_result(args.f3_root, seed, 'F3_AlignedPeriodicityFusionHead', F3),
    ]
    model = _load_model_from_state(f2_dir, device, initial_state)
    model, training = _train_loss_model(model, 'L2_Focal_RawSaw', roles, weak6, seed, device, args, root / VARIANT)
    logits = {role: _predict(model, values['signals'], device, args.eval_batch_size) for role, values in roles.items()}
    rows.append(_evaluate(VARIANT, _route_rawsaw(logits, raw_scores), roles, weak6, root / VARIANT, args))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    history = _late_history_summary(root / VARIANT / 'history.csv')
    payload = {
        'seed': int(seed), 'status': 'completed', 'data_sha256': data_sha,
        'role_indices_sha256': _sha256(role_path), 'f2_checkpoint_sha256': _sha256(f2_checkpoint),
        'rawsaw_checkpoint_sha256': _sha256(raw_checkpoint),
        'training': training, 'history_summary': history,
        'focal72_checkpoint_sha256': _sha256(root / VARIANT / 'best_model.pth'),
        'variants': rows, 'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False,
    }
    _write_json(receipt, payload)
    return payload


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError('development-v3 SHA mismatch')
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('test-v3 is not sealed')
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / 'experiment_config.json', {
        'experiment': 'RouteA_DevelopmentV3_Focal72CosineConvergenceAudit', 'seeds': args.seeds,
        'epochs': args.epochs, 'lr_schedule': args.lr_schedule, 'minimum_lr_ratio': args.minimum_lr_ratio,
        'patience': args.patience, 'early_stopping_disabled_by_patience_equals_epochs': args.patience >= args.epochs,
        'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False,
    })
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in args.seeds]
    names = (FOCAL36, H1, F3, VARIANT)
    aggregate = _aggregate(rows, names)
    versus_focal36 = _comparison(rows, VARIANT, FOCAL36, 20803000)
    versus_h1 = _comparison(rows, VARIANT, H1, 20804000)
    convergence = {
        'best_epochs': [int(row['training']['best_epoch']) for row in rows],
        'best_epoch_at_cap_count': int(sum(int(row['training']['best_epoch']) == args.epochs for row in rows)),
        'all_runs_completed_epochs': all(int(row['training']['epochs_ran']) == args.epochs for row in rows),
        'history_summary': {str(row['seed']): row['history_summary'] for row in rows},
    }
    result = {
        'status': 'completed', 'experiment': 'RouteA_DevelopmentV3_Focal72CosineConvergenceAudit',
        'evidence_role': 'development_v3_fixed_baseline_convergence_audit', 'seeds': args.seeds,
        'data_sha256': data_sha, 'aggregate': aggregate, 'focal72_vs_focal36': versus_focal36,
        'focal72_vs_h1': versus_h1, 'convergence': convergence,
        'selected_or_promoted_model': False, 'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False,
    }
    output = args.output_root / 'route_a_development_v3_focal72_cosine_convergence_summary.json'
    _write_json(output, result)
    print(json.dumps({'status': 'completed', 'convergence': convergence, 'summary': str(output)}, ensure_ascii=False, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/student_a/six_class_development_v3.yaml')
    parser.add_argument('--f2-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_dfms_iq_structure_screen')
    parser.add_argument('--c6-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_development_v3_label_selective_projection')
    parser.add_argument('--unified-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3')
    parser.add_argument('--focal36-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_development_v3_loss_convergence_audit')
    parser.add_argument('--head-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines')
    parser.add_argument('--f3-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3')
    parser.add_argument('--test-v3-manifest', type=Path, default=PROJECT_ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs/tapc/route_a_development_v3_focal72_cosine_convergence')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument('--epochs', type=int, default=72)
    parser.add_argument('--patience', type=int, default=72)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--io-batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--raw-batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--representation-lr-scale', type=float, default=.1)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='cosine')
    parser.add_argument('--minimum-lr-ratio', type=float, default=.01)
    parser.add_argument('--saw-loss-weight', type=float, default=.2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gradient-clip', type=float, default=5.0)
    parser.add_argument('--max-negative-fpr', type=float, default=.20)
    parser.add_argument('--weak-threshold-db', type=float, default=10.0)
    parser.add_argument('--raw-hidden-channels', type=int, default=16)
    parser.add_argument('--raw-topk', type=int, default=16)
    parser.add_argument('--raw-dropout', type=float, default=.1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-epochs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    return parser


if __name__ == '__main__':
    arguments = build_parser().parse_args()
    if arguments.smoke:
        arguments.epochs = arguments.smoke_epochs
        arguments.patience = arguments.smoke_epochs
    run(arguments)
