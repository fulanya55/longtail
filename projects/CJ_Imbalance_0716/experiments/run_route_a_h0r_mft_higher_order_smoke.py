"""No-fit, no-data-role H0r smoke for MFT count and phase-surrogate contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cjlt.models.mft_higher_order import circular_autocorrelation, dft_phase_surrogate, segmented_bicoherence


def lfm_mft(seed: int, count: int) -> np.ndarray:
    rng=np.random.default_rng(seed); size=2048; time=np.arange(1000)/10e6; base=np.exp(1j*np.pi*(5e6/100e-6)*time*time); output=np.zeros(size,dtype=np.complex128)
    for delay, amplitude in zip(rng.choice(np.arange(300,501),size=count,replace=False), .5+rng.random(count)):
        output[delay:delay+base.size] += amplitude*base
    return output


def run() -> dict[str, object]:
    magnitude_error=[]; autocorrelation_error=[]; candidate=[]; control=[]
    for item in range(10):
        value=lfm_mft(3_700_000_000+item, 3 if item%2==0 else 8); surrogate=dft_phase_surrogate(value,seed=3_710_000_000+item)
        magnitude_error.append(float(np.max(np.abs(np.abs(np.fft.fft(value))-np.abs(np.fft.fft(surrogate))))))
        autocorrelation_error.append(float(np.max(np.abs(circular_autocorrelation(value)-circular_autocorrelation(surrogate)))))
        candidate.append(segmented_bicoherence(value)); control.append(segmented_bicoherence(surrogate))
    result={"status":"completed_smoke_only","experiment":"RouteA_H0r_MFTHigherOrder","sample_count":10,"statistic":"fixed_segmented_bicoherence_bins_7_11","surrogate":"full_record_DFT_random_phase_exact_magnitude_and_circular_autocorrelation_control","maximum_dft_magnitude_error":max(magnitude_error),"maximum_circular_autocorrelation_error":max(autocorrelation_error),"candidate_mean":float(np.mean(candidate)),"surrogate_mean":float(np.mean(control)),"trained_parameters":0,"formal_h0_authorized":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    output=ROOT/"outputs/tapc/route_a_h0r_mft_higher_order_smoke/summary.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False)); return result


if __name__ == "__main__": run()
