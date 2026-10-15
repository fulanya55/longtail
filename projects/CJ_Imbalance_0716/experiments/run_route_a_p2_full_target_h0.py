"""Run P2's fixed no-fit all-target physical-observability gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_route_a_p1_observability_adequate_h0 as p1  # noqa: E402


BLOCK_SIZE = 25
BLOCK_PERMUTATION = np.random.default_rng(20260803).permutation(p1.LENGTH // BLOCK_SIZE)


def _block_shuffle(signal: np.ndarray) -> np.ndarray:
    if signal.size % BLOCK_SIZE:
        raise ValueError("P2 record length must be divisible by block size")
    return np.asarray(signal).reshape(-1, BLOCK_SIZE)[BLOCK_PERMUTATION].reshape(-1)


def _isrj_joint_periodicity_score(signal: np.ndarray) -> float:
    z = np.asarray(signal, dtype=np.complex128)
    z = z / np.sqrt(max(float(np.mean(np.abs(z) ** 2)), np.finfo(np.float64).eps))
    transformed = np.fft.fft(z, n=2 * z.size)
    complex_acf = np.abs(np.fft.ifft(np.abs(transformed) ** 2))[: z.size]
    complex_acf /= max(float(complex_acf[0]), np.finfo(np.float64).eps)
    envelope = np.abs(z); envelope = envelope - envelope.mean()
    envelope_fft = np.fft.fft(envelope, n=2 * envelope.size)
    envelope_acf = np.abs(np.fft.ifft(np.abs(envelope_fft) ** 2))[: envelope.size]
    envelope_acf /= max(float(envelope_acf[0]), np.finfo(np.float64).eps)
    return float(.5 * (np.max(complex_acf[200:601]) + np.max(envelope_acf[200:601])))


def _score_rows(iq: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    true = {name: np.empty(iq.shape[0], dtype=np.float64) for name in p1.TARGETS.values()}
    control = {name: np.empty(iq.shape[0], dtype=np.float64) for name in p1.TARGETS.values()}
    for row, signal in enumerate(iq):
        true["MFT"][row] = p1._mft_score(signal, p1.K); control["MFT"][row] = p1._mft_score(signal, .83 * p1.K)
        true["VFT"][row], control["VFT"][row] = p1._vft_scores(signal)
        true["ISRJ"][row] = _isrj_joint_periodicity_score(signal)
        control["ISRJ"][row] = _isrj_joint_periodicity_score(_block_shuffle(signal))
    return true, control


def run(args: argparse.Namespace) -> dict:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 must remain sealed")
    iq, labels, pair, receiver, target, arm = p1._read(args.data_path)
    true, control = _score_rows(iq); triplets = p1._triplets(pair, receiver, target, arm, labels)
    expected = 3 if args.smoke else 300
    if any(rows[0].size != expected for rows in triplets.values()):
        raise ValueError(f"P2 expected {expected} identities per target")
    finite = bool(all(np.all(np.isfinite(value)) for value in (*true.values(), *control.values())))
    if args.smoke:
        result = {"status":"completed_smoke_score_only","experiment":"RouteA_P2_FullTargetObservability","view_count":int(iq.shape[0]),"scores_finite":finite,"metrics_reported":False,"gate_evaluated":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    else:
        values={}; outcomes=[]
        for name,(absent_rows,weak_rows,strong_rows) in triplets.items():
            metric=p1._metrics(true[name][absent_rows],true[name][weak_rows],true[name][strong_rows],control[name][absent_rows],control[name][weak_rows])
            bootstrap=p1._bootstrap(true[name][absent_rows],true[name][weak_rows],true[name][strong_rows],control[name][absent_rows],control[name][weak_rows],args.bootstrap_draws)
            checks={"true_auroc_ge_0p75":metric["true_auroc"]>=.75,"true_minus_control_auroc_ge_0p10":metric["true_minus_control_auroc"]>=.10,"delta_ci_lower_gt_zero":bootstrap["true_minus_control_auroc_95_ci"][0]>0,"paired_weak_gt_absent_fraction_ge_0p65":metric["paired_weak_gt_absent_fraction"]>=.65,"strong_auroc_not_below_weak_by_more_than_0p02":metric["strong_vs_absent_auroc"]>=metric["true_auroc"]-.02}
            values[name]={"identity_count":int(absent_rows.size),**metric,"bootstrap":{"draws":args.bootstrap_draws,**bootstrap},"checks":checks,"passed":bool(all(checks.values()))}; outcomes.append(bool(all(checks.values())))
        result={"status":"completed_no_fit_observable","experiment":"RouteA_P2_FullTargetObservability","evidence_role":"external_protocol_h0_development_only","view_count":int(iq.shape[0]),"fixed_statistics":{"mft_bands_hz":p1.MFT_BANDS_HZ,"vft_band_hz":p1.VFT_BAND_HZ,"pulse_permutation":p1.PULSE_PERMUTATION.tolist(),"isrj_acf_lags":[200,600],"isrj_block_size":BLOCK_SIZE,"isrj_block_permutation_seed":20260803},"trained_parameters":0,"metrics_by_target":values,"gate":{"passed":bool(finite and all(outcomes)),"all_scores_finite":finite},"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-path",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--bootstrap-draws",type=int,default=5000); parser.add_argument("--test-v3-manifest",type=Path,default=ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    result=run(parser.parse_args()); print(json.dumps({"status":result["status"],"gate":result.get("gate")},ensure_ascii=False,indent=2))
