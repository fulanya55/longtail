"""Exact frozen B0/B2 replication on farther ISRJ parameters; never fits external data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_isrj_parameter_ood_v1_frozen import (  # noqa: E402
    METRICS, VARIANTS, _mean_ci, _run_seed, _vm,
)


def run(args):
    # Reuse only the audited reader and exact frozen inference implementation.
    from experiments import run_route_a_isrj_parameter_ood_v1_frozen as base

    audit = json.loads(args.audit.read_text(encoding='utf8'))
    seal = json.loads(args.test_v3_manifest.read_text(encoding='utf8'))
    if not audit['identity_audit_passed'] or not all(audit['contracts'].values()):
        raise ValueError('H0g data audit failed')
    if any(seal['test_v3'][key] for key in ('access_authorized', 'analysis_allowed', 'used')):
        raise ValueError('test-v3 is not sealed')
    data_sha = audit['sha256']
    labels = base.read_labels(args.data_path)
    values, _ = base.read_parameters(args.data_path)
    weak = base.compute_weak_component_mask(labels, base._component_power(values), args.weak_threshold_db)
    signals = base._load_signals(args.data_path, base.np.arange(len(labels)), args.io_batch_size)
    with base.h5py.File(args.data_path, 'r') as handle:
        receiver = base.np.asarray(handle['receiver_noise_seed']).reshape(-1).astype(base.np.int64)
    device = base.torch.device(args.device or ('cuda' if base.torch.cuda.is_available() else 'cpu'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    # DATA_SHA in the legacy inference receipt is a source identifier, not a
    # learned artifact. Patch it only for this call and restore immediately.
    old_sha = base.DATA_SHA
    try:
        base.DATA_SHA = data_sha
        rows = [base._run_seed(args, labels, values, weak, signals, receiver, seed, device) for seed in args.seeds]
    finally:
        base.DATA_SHA = old_sha
    aggregate = {
        variant: {name: _mean_ci([fn(_vm(row)[variant]['metrics']) for row in rows], 20482000 + vi * 100 + mi)
                  for mi, (name, fn) in enumerate(METRICS.items())}
        for vi, variant in enumerate(VARIANTS)
    }
    per_seed = []
    for row in rows:
        b0 = _vm(row)[VARIANTS[0]]['metrics']
        b2 = _vm(row)[VARIANTS[1]]['metrics']
        per_seed.append({'seed': row['seed'], **{name: fn(b2) - fn(b0) for name, fn in METRICS.items()}})
    delta = {name: _mean_ci([row[name] for row in per_seed], 20483000 + i) for i, name in enumerate(METRICS)}
    delta['positive_weak_recall_seeds'] = sum(row['isrj_weak_recall'] > 0 for row in per_seed)
    delta['positive_weak_auroc_seeds'] = sum(row['isrj_weak_auroc'] > 0 for row in per_seed)
    checks = {
        'isrj_aupr_positive': delta['isrj_aupr']['mean'] > 0,
        'weak_recall_ge_0p030': delta['isrj_weak_recall']['mean'] >= 0.030,
        'weak_auroc_positive': delta['isrj_weak_auroc']['mean'] > 0,
        'four_of_five_recall': delta['positive_weak_recall_seeds'] >= 4,
        'four_of_five_auroc': delta['positive_weak_auroc_seeds'] >= 4,
        'b2_fpr_le_0p25': aggregate[VARIANTS[1]]['isrj_fpr']['mean'] <= 0.25,
        'macro_aupr_nonnegative': delta['macro_aupr']['mean'] >= 0,
        'sawfm_exact': abs(delta['sawfm_aupr']['mean']) < 1e-10 and abs(delta['sawfm_weak_recall']['mean']) < 1e-10,
        'no_external_refit': all(not row['external_refit'] for row in rows),
        'test_v3_sealed': all(not row['test_v3_used'] for row in rows),
    }
    result = {
        'status': 'completed', 'experiment': args.experiment,
        'evidence_role': args.evidence_role, 'seeds': args.seeds,
        'external_data_sha256': data_sha, 'data_audit_path': str(args.audit),
        'frozen_source_runner': str(Path(base.__file__).resolve()), 'aggregate': aggregate,
        'rows': rows,
        'b2_vs_b0': {'per_seed': per_seed, 'aggregate': delta},
        'external_gate': {'passed': bool(all(checks.values())), 'checks': checks},
        'external_refit': False, 'locked_test_used': False, 'test_v2_used': False,
        'test_v3_used': False, 'test_h0g_used': False,
    }
    output = args.output_root / args.summary_name
    base._write_json(output, result)
    print(json.dumps({'status': result['status'], 'gate': result['external_gate'], 'summary': str(output)}, ensure_ascii=False, indent=2))
    return result


def build_parser():
    base = PROJECT_ROOT / 'outputs/tapc'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_ood_h0g\closedset_6class_train.mat'))
    parser.add_argument('--development-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat'))
    parser.add_argument('--audit', type=Path, default=base / 'route_a_h0g_isrj_parameter_ood_v2_generation/identity_parameter_audit.json')
    parser.add_argument('--test-v3-manifest', type=Path, default=PROJECT_ROOT / 'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    parser.add_argument('--f2-root', type=Path, default=base / 'route_a_dfms_iq_structure_screen')
    parser.add_argument('--c6-root', type=Path, default=base / 'route_a_development_v3_label_selective_projection')
    parser.add_argument('--unified-root', type=Path, default=base / 'route_a_unified_rawsaw_vtse_vbf_v3')
    parser.add_argument('--convergence-root', type=Path, default=base / 'route_a_development_v3_loss_convergence_audit')
    parser.add_argument('--head-root', type=Path, default=base / 'route_a_development_v3_frozen_head_capacity_baselines')
    parser.add_argument('--increment-root', type=Path, default=base / 'route_a_development_v3_h1_periodicity_increment')
    parser.add_argument('--old-f3-root', type=Path, default=base / 'route_a_isrj_frozen_score_fusion_head_v3')
    parser.add_argument('--periodicity-root', type=Path, default=base / 'route_a_isrj_joint_periodicity_expert_v3')
    parser.add_argument('--output-root', type=Path, default=base / 'route_a_h0g_isrj_parameter_ood_v2_frozen')
    parser.add_argument('--experiment', default='RouteA_H0g_ISRJParameterOODv2_Frozen')
    parser.add_argument('--evidence-role', default='fresh_far_parameter_ood_frozen_replication')
    parser.add_argument('--summary-name', default='route_a_h0g_isrj_parameter_ood_v2_frozen_summary.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument('--io-batch-size', type=int, default=128)
    parser.add_argument('--model-batch-size', type=int, default=128)
    parser.add_argument('--periodicity-batch-size', type=int, default=64)
    parser.add_argument('--raw-batch-size', type=int, default=64)
    parser.add_argument('--raw-hidden-channels', type=int, default=16)
    parser.add_argument('--raw-topk', type=int, default=16)
    parser.add_argument('--raw-dropout', type=float, default=0.1)
    parser.add_argument('--weak-threshold-db', type=float, default=10.0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--resume', action='store_true')
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == '__main__':
    run(parse_args())
