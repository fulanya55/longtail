"""Frozen B0/B2 inference on the H0s joint parameter-plus-noise source."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_route_a_h0g_isrj_parameter_ood_v2_frozen as frozen  # noqa: E402


def build_parser():
    parser = frozen.build_parser()
    parser.set_defaults(
        data_path=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1\closedset_6class_train.mat'),
        audit=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_generation/identity_parameter_noise_audit.json',
        output_root=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_frozen',
        experiment='RouteA_H0s_ISRJCombinedParameterNoiseFrozen',
        evidence_role='fresh_joint_parameter_receiver_noise_frozen_replication',
        summary_name='route_a_h0s_isrj_combined_frozen_summary.json',
    )
    return parser


def run(args):
    """Delegate inference to the exact audited H0g frozen implementation."""
    result = frozen.run(args)
    result['experiment'] = 'RouteA_H0s_ISRJCombinedParameterNoiseFrozen'
    result['evidence_role'] = 'fresh_joint_parameter_receiver_noise_frozen_replication'
    result['joint_shift'] = {
        'isrj_sample_width_s': [25e-6, 30e-6],
        'isrj_num_replicas': 9,
        'receiver_noise': 'proper_complex_ar1_rho_0p80',
    }
    result.pop('test_h0g_used', None)
    result['test_h0s_used'] = False
    output = args.output_root / args.summary_name
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'gate': result['external_gate'], 'summary': str(output)}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    run(build_parser().parse_args())
