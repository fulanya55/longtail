"""GPU shared FFT/autocorrelation/spectrum front-end for MFT and ISRJ."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def _pool(sequence: torch.Tensor, bins: int) -> torch.Tensor:
    usable = sequence.shape[1] // bins * bins
    return sequence[:, :usable].reshape(len(sequence), bins, -1).amax(dim=2)


@torch.no_grad()
def shared_feature_batch(iq: torch.Tensor) -> dict[str, torch.Tensor]:
    if iq.ndim != 3 or iq.shape[1:] != (2, 3000):
        raise ValueError("iq must have shape [B,2,3000]")
    signal = torch.complex(iq[:, 0], iq[:, 1])
    signal = signal / torch.sqrt(signal.abs().square().mean(dim=1, keepdim=True)).clamp_min(1e-8)
    transformed = torch.fft.fft(signal, n=8192, dim=1)
    complex_acf = torch.fft.ifft(transformed.abs().square(), dim=1).abs()[:, 1:1201]
    complex_acf = complex_acf / complex_acf[:, :1].clamp_min(1e-8)
    envelope = signal.abs()
    envelope = envelope - envelope.mean(dim=1, keepdim=True)
    envelope_fft = torch.fft.fft(envelope, n=8192, dim=1)
    envelope_acf = torch.fft.ifft(envelope_fft.abs().square(), dim=1).abs()[:, 1:1201]
    envelope_acf = envelope_acf / envelope_acf[:, :1].clamp_min(1e-8)
    spectrum = torch.fft.fftshift(torch.fft.fft(signal, dim=1), dim=1).abs()
    spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1e-8)
    complex_bins = _pool(complex_acf, 120)
    envelope_bins = _pool(envelope_acf, 120)
    spectrum_bins = spectrum.reshape(len(signal), 60, 50).sum(dim=2)
    delay = complex_acf[:, 249:1000]
    periodic = envelope_acf[:, 39:600]
    delay_peak = delay.amax(dim=1, keepdim=True)
    delay_position = delay.argmax(dim=1, keepdim=True).float() / delay.shape[1]
    periodic_peak = periodic.amax(dim=1, keepdim=True)
    periodic_position = periodic.argmax(dim=1, keepdim=True).float() / periodic.shape[1]
    complex_top = delay.topk(8, dim=1).values.sort(dim=1).values
    envelope_top = periodic.topk(8, dim=1).values.sort(dim=1).values
    mft = torch.cat((complex_bins, spectrum_bins, delay_peak, delay_position, complex_top), dim=1)
    isrj = torch.cat((complex_bins, envelope_bins, spectrum_bins,
                      periodic_peak, periodic_position, envelope_top), dim=1)
    combined = torch.cat((complex_bins, envelope_bins, spectrum_bins,
                          delay_peak, delay_position, periodic_peak, periodic_position,
                          complex_top, envelope_top), dim=1)
    return {"mft_delay": mft.float(), "isrj_periodicity": isrj.float(), "combined": combined.float()}


def extract_shared_features(
    mat_path: Path, indices: np.ndarray, device: torch.device, batch_size: int,
) -> dict[str, np.ndarray]:
    output = {"mft_delay": [], "isrj_periodicity": [], "combined": []}
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    with h5py.File(mat_path, "r") as handle:
        for start in range(0, len(sorted_indices), batch_size):
            selected = sorted_indices[start:start + batch_size]
            raw = np.asarray(handle["data_legacy"][:, selected], dtype=np.float32).T.reshape(-1, 2, 3000)
            features = shared_feature_batch(torch.from_numpy(raw).to(device))
            for key in output:
                output[key].append(features[key].cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in output.items()}
