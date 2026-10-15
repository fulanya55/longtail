"""FFT-equivalent implementation of the frozen E5 VFT matched filter."""

from __future__ import annotations

import numpy as np
import torch
import h5py

from cj0724.vft_physics_screen import (
    CHIRP_RATE_HZ_PER_S,
    FS_HZ,
    PULSE_SAMPLES,
    feature_matrix,
    build_vft_filter_bank,
)


def build_complex_templates(frequencies_hz: np.ndarray, device: torch.device) -> torch.Tensor:
    time = torch.arange(PULSE_SAMPLES, device=device, dtype=torch.float32) / FS_HZ
    frequencies = torch.as_tensor(frequencies_hz, device=device, dtype=torch.float32)
    phase = (
        torch.pi * CHIRP_RATE_HZ_PER_S * time[None, :].square()
        + 2.0 * torch.pi * frequencies[:, None] * time[None, :]
    )
    return torch.polar(torch.ones_like(phase), phase)


def fft_matched_filter_peaks(
    iq: torch.Tensor, frequencies_hz: np.ndarray, nfft: int = 4096,
    transformed_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the same valid cross-correlation peaks as the E5 direct convolution."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 3000):
        raise ValueError("iq must have shape [B, 2, 3000]")
    required = iq.shape[-1] + PULSE_SAMPLES - 1
    if nfft < required:
        raise ValueError(f"nfft must be at least {required}")
    if transformed_weights is None:
        weights, _ = build_vft_filter_bank(frequencies_hz, iq.device)
        transformed_weights = torch.fft.rfft(weights.flip(-1), n=nfft, dim=-1)
    transformed_signal = torch.fft.rfft(iq, n=nfft, dim=-1)
    convolution = torch.fft.irfft(
        torch.einsum("bck,ock->bok", transformed_signal, transformed_weights),
        n=nfft, dim=-1,
    )
    valid_length = iq.shape[-1] - PULSE_SAMPLES + 1
    correlation = convolution[..., PULSE_SAMPLES - 1:PULSE_SAMPLES - 1 + valid_length]
    count = len(frequencies_hz)
    real, imag = correlation[:, :count], correlation[:, count:]
    instantaneous_energy = iq.square().sum(dim=1)
    prefix = torch.nn.functional.pad(instantaneous_energy.cumsum(dim=-1), (1, 0))
    local_energy = (prefix[:, PULSE_SAMPLES:] - prefix[:, :-PULSE_SAMPLES]).clamp_min(1.0e-8)
    normalized = torch.sqrt(real.square() + imag.square() + 1.0e-12) / torch.sqrt(
        local_energy[:, None, :] * float(PULSE_SAMPLES)
    )
    return normalized.max(dim=2)


def fft_feature_matrix(iq: torch.Tensor, frequencies_hz: np.ndarray) -> np.ndarray:
    peaks, positions = fft_matched_filter_peaks(iq, frequencies_hz)
    return feature_matrix(peaks.cpu().numpy(), positions.cpu().numpy(), frequencies_hz)


def extract_features_fft(
    mat_path, indices: np.ndarray, frequencies_hz: np.ndarray,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
    features = []
    weights, _ = build_vft_filter_bank(frequencies_hz, device)
    transformed_weights = torch.fft.rfft(weights.flip(-1), n=4096, dim=-1)
    with h5py.File(mat_path, "r") as handle:
        for start in range(0, len(sorted_indices), batch_size):
            selected = sorted_indices[start:start + batch_size]
            raw = np.asarray(handle["data_legacy"][:, selected], dtype=np.float32).T
            iq = raw.reshape(-1, 2, 3000)
            rms = np.sqrt(np.mean(np.square(iq).sum(axis=1), axis=1, keepdims=True))
            iq = iq / np.maximum(rms[:, None], 1.0e-8)
            with torch.no_grad():
                tensor = torch.from_numpy(iq).to(device)
                peaks, positions = fft_matched_filter_peaks(
                    tensor, frequencies_hz, transformed_weights=transformed_weights
                )
                features.append(feature_matrix(
                    peaks.cpu().numpy(), positions.cpu().numpy(), frequencies_hz
                ))
    return np.concatenate(features)
