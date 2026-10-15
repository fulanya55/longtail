"""Fixed H0r higher-order MFT smoke primitives; no learned parameters."""
from __future__ import annotations

import numpy as np


def dft_phase_surrogate(signal: np.ndarray, *, seed: int) -> np.ndarray:
    """Preserve DFT magnitudes while destroying deterministic Fourier phases."""
    value = np.asarray(signal, dtype=np.complex128)
    if value.ndim != 1 or value.size < 16:
        raise ValueError("signal must be a one-dimensional record of at least 16 samples")
    spectrum = np.fft.fft(value)
    phase = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=value.size)
    return np.fft.ifft(np.abs(spectrum) * np.exp(1j * phase))


def circular_autocorrelation(signal: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(np.asarray(signal, dtype=np.complex128))
    return np.fft.ifft(np.abs(spectrum) ** 2)


def segmented_bicoherence(signal: np.ndarray, *, segments: int = 8, bins: tuple[int, int] = (7, 11)) -> float:
    """Fixed normalized third-order coupling over contiguous equal windows."""
    value = np.asarray(signal, dtype=np.complex128)
    length = value.size // int(segments)
    if length <= max(bins) * 2 or length < 16:
        raise ValueError("record is too short for requested segments and bins")
    views = value[: length * segments].reshape(segments, length)
    transform = np.fft.fft(views, axis=1)
    first, second = bins; third = (first + second) % length
    product = transform[:, first] * transform[:, second] * np.conj(transform[:, third])
    denominator = np.sqrt(np.mean(np.abs(transform[:, first] * transform[:, second]) ** 2) * np.mean(np.abs(transform[:, third]) ** 2))
    return float(np.abs(np.mean(product)) / max(float(denominator), np.finfo(np.float64).eps))
