"""Post-hoc fixed-score uncertainty diagnostic for preregistered H1j.

This script neither trains nor selects a model.  It reconstructs the two
already-written seed-2039 readouts, resamples only the fixed evaluation rows,
and records an ordinary class-stratified paired bootstrap for the difference
between candidate and control ranking metrics.  Optimizer-seed variability is
reported separately because rank metrics are invariant to the tiny changes in
the one-dimensional monotone heads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from cjlt.models.vft_dechirped_doppler import fixed_phase_permutation, lfm_reference_phase
from experiments.run_route_a_development_v2_shared_gradient import _component_power
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals
from experiments.run_route_a_h1j_vft_dechirped_composite import (
    VFT_INDEX, _matched_view, _scores, _source,
)


def _probability(score: np.ndarray, head: dict) -> np.ndarray:
    logits = ((score - float(head['mean'])) / float(head['std'])) * float(head['weight']) + float(head['bias'])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def bootstrap_paired_delta(y: np.ndarray, candidate: np.ndarray, control: np.ndarray, *, draws: int, seed: int) -> dict:
    """Class-stratified resampling of fixed rows; each draw scores both arms."""
    y = np.asarray(y, dtype=np.uint8)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError('need at least two fixed rows per class')
    generator = np.random.default_rng(int(seed))
    values = {'auroc': np.empty(int(draws)), 'aupr': np.empty(int(draws))}
    for draw in range(int(draws)):
        index = np.concatenate((generator.choice(positive, len(positive), replace=True), generator.choice(negative, len(negative), replace=True)))
        values['auroc'][draw] = roc_auc_score(y[index], candidate[index]) - roc_auc_score(y[index], control[index])
        values['aupr'][draw] = average_precision_score(y[index], candidate[index]) - average_precision_score(y[index], control[index])
    return {name: {'point': float((roc_auc_score(y, candidate) - roc_auc_score(y, control)) if name == 'auroc' else (average_precision_score(y, candidate) - average_precision_score(y, control))),
                   'bootstrap_95_ci': [float(np.quantile(value, .025)), float(np.quantile(value, .975))],
                   'bootstrap_mean': float(value.mean()), 'draws': int(draws), 'resampling': 'fixed evaluation rows, class-stratified, paired arms'} for name, value in values.items()}


def run() -> dict:
    output_root = PROJECT_ROOT / 'outputs/tapc/route_a_h1j_vft_dechirped_composite'
    seed_receipt = json.loads((output_root / 'seed_2039/seed_summary.json').read_text(encoding='utf8'))
    summary = json.loads((output_root / 'route_a_h1j_vft_dechirped_composite_summary.json').read_text(encoding='utf8'))
    if not summary['gate']['passed'] or any(summary[key] for key in ('locked_test_used', 'test_v2_used', 'test_v3_used', 'test_h1j_used')):
        raise ValueError('H1j state is not the expected sealed-source completed state')
    labels = read_labels(_source())
    params, _ = read_parameters(_source())
    weak = compute_weak_component_mask(labels, _component_power(params), 10.0)
    roles = np.load(PROJECT_ROOT / 'outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/role_indices.npz')
    evaluation_ids = np.asarray(roles['evaluation'], dtype=np.int64)
    keep, y, counts = _matched_view(labels[evaluation_ids], weak[evaluation_ids], evaluation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signals = _load_signals(_source(), np.arange(len(labels)), 128)
    reference = lfm_reference_phase(1000, device=device)
    candidate_score = _scores(signals, reference, device, 128)[evaluation_ids][keep]
    control_score = _scores(signals, fixed_phase_permutation(reference), device, 128)[evaluation_ids][keep]
    candidate = _probability(candidate_score, seed_receipt['arms']['CandidateCorrectDechirp']['head'])
    control = _probability(control_score, seed_receipt['arms']['ControlPermutedPhase']['head'])
    metrics = bootstrap_paired_delta(y, candidate, control, draws=10000, seed=20802111)
    payload = {
        'status': 'completed_fixed_score_inference_only', 'experiment': 'RouteA_H1j_VFTDechirpedCompositeSource',
        'purpose': 'post-hoc evaluation-row uncertainty diagnostic; does not train, select, calibrate, or threshold',
        'reference_seed': 2039, 'evaluation_weak_matched_counts': counts, 'metrics_candidate_minus_control': metrics,
        'optimizer_seed_rank_invariant_notice': 'The five fixed one-dimensional heads preserve ranking, so their near-zero metric variation is not a replacement for sample-level uncertainty.',
        'no_model_change': True, 'no_threshold_or_calibration': True,
        'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False, 'test_h1j_used': False,
    }
    path = output_root / 'h1j_fixed_score_stratified_bootstrap.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == '__main__':
    run()
