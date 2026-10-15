"""Formal zero-fit H0r MFT bicoherence test with an exact second-order control."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cjlt.models.mft_higher_order import circular_autocorrelation, dft_phase_surrogate, segmented_bicoherence


def _record(seed: int, count: int) -> np.ndarray:
    rng=np.random.default_rng(seed); size=2048; time=np.arange(1000)/10e6; base=np.exp(1j*np.pi*(5e6/100e-6)*time*time); value=np.zeros(size,dtype=np.complex128)
    delays=rng.choice(np.arange(300,501),size=8,replace=False); amplitude=.5+rng.random(8)
    for delay,weight in zip(delays[:count],amplitude[:count]): value[delay:delay+base.size]+=weight*base
    value*=np.sqrt(100.0/max(np.mean(np.abs(value)**2),1e-12)); noise_rng=np.random.default_rng((seed+2*2_246_822_519)%(2**32)); return value+(noise_rng.standard_normal(size)+1j*noise_rng.standard_normal(size))/np.sqrt(2)


def _ci(low,high,clow,chigh):
    rng=np.random.default_rng(20260803); y=np.r_[np.zeros(len(low)),np.ones(len(high))]; values=[]
    for _ in range(5000):
        i=rng.integers(0,len(low),len(low)); values.append(roc_auc_score(y,np.r_[high[i]*0+low[i],high[i]])-roc_auc_score(y,np.r_[clow[i],chigh[i]]))
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]


def run() -> dict[str, object]:
    low=[]; high=[]; clow=[]; chigh=[]; mag=[]; ac=[]
    for index in range(640):
        for count,score,control in ((3,low,clow),(8,high,chigh)):
            value=_record(3_800_000_000+index,count); surrogate=dft_phase_surrogate(value,seed=3_810_000_000+2*index+(count==8)); score.append(segmented_bicoherence(value)); control.append(segmented_bicoherence(surrogate))
            mag.append(float(np.max(np.abs(np.abs(np.fft.fft(value))-np.abs(np.fft.fft(surrogate))))/max(np.max(np.abs(np.fft.fft(value))),1e-12))); ac.append(float(np.max(np.abs(circular_autocorrelation(value)-circular_autocorrelation(surrogate)))/max(np.max(np.abs(circular_autocorrelation(value))),1e-12)))
    low,high,clow,chigh=map(np.asarray,(low,high,clow,chigh)); y=np.r_[np.zeros(640),np.ones(640)]; au=float(roc_auc_score(y,np.r_[low,high])); control=float(roc_auc_score(y,np.r_[clow,chigh])); ci=_ci(low,high,clow,chigh); checks={"candidate_auroc_ge_0p65":au>=.65,"candidate_minus_control_auroc_ge_0p10":au-control>=.10,"paired_bootstrap_lower_gt_zero":ci[0]>0,"paired_8_gt_3_rate_ge_0p65":float(np.mean(high>low))>=.65,"dft_magnitude_relative_error_le_1e8":max(mag)<=1e-8,"circular_autocorrelation_relative_error_le_1e8":max(ac)<=1e-8,"trained_parameters_zero":True,"test_v3_still_unused":True}
    result={"status":"completed_no_fit_h0","experiment":"RouteA_H0r_MFTHigherOrder","identity_pair_count":640,"source_seed_range":[3800000000,3800000639],"statistic":"segmented_bicoherence_segments8_bins7_11","control":"DFT_random_phase_exact_magnitude_circular_autocorrelation","candidate_auroc":au,"control_auroc":control,"candidate_minus_control_auroc":au-control,"bootstrap_95_ci":ci,"paired_8_greater_3_fraction":float(np.mean(high>low)),"maximum_dft_magnitude_relative_error":max(mag),"maximum_circular_autocorrelation_relative_error":max(ac),"gate":{"passed":bool(all(checks.values())),"checks":checks},"trained_parameters":0,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}; output=ROOT/"outputs/tapc/route_a_h0r_mft_higher_order/summary.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False)); return result


if __name__ == "__main__": run()
