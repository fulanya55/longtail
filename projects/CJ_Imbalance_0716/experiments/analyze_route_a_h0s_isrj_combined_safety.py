"""Non-selection FPR safety report for the completed H0s frozen replication."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_route_a_h0h_ar1_frozen_safety as safety  # noqa: E402


def run(args):
    result = safety.run(args)
    result['experiment'] = 'RouteA_H0s_ISRJCombinedParameterNoiseFrozen'
    result['evidence_role'] = 'post_completed_nonselection_joint_shift_safety_description'
    result['test_h0s_used'] = False
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result['aggregate'], ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1\closedset_6class_train.mat'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_frozen')
    parser.add_argument('--summary', type=Path, default=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_frozen/route_a_h0s_isrj_combined_frozen_summary.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tapc/route_a_h0s_isrj_combined_frozen/nonselection_safety_report.json')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
