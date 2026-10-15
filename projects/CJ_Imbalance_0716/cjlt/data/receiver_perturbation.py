"""Controlled receiver non-idealities for simulation-domain robustness tests."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ReceiverPerturbationConfig:
    iq_gain_imbalance_db: float = 0.0
    quadrature_error_deg: float = 0.0
    phase_noise_std_rad: float = 0.0
    carrier_frequency_offset_cycles: float = 0.0
    amplitude_scale: float = 1.0
    sample_shift: int = 0
    local_mask_fraction: float = 0.0

    def to_dict(self):
        return asdict(self)


def apply_receiver_perturbation(signal, config: ReceiverPerturbationConfig):
    values = torch.as_tensor(signal).float()
    squeezed = values.ndim == 2
    if squeezed:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("signal must have shape [2, L] or [B, 2, L]")
    output = values.clone()
    length = output.shape[-1]

    gain_ratio = 10.0 ** (float(config.iq_gain_imbalance_db) / 20.0)
    real = output[:, 0] * float(config.amplitude_scale) * gain_ratio
    imag = output[:, 1] * float(config.amplitude_scale) / max(gain_ratio, 1e-12)

    quadrature = math.radians(float(config.quadrature_error_deg))
    if quadrature:
        imag = imag * math.cos(quadrature) + real * math.sin(quadrature)

    phase = torch.zeros((len(output), length), dtype=output.dtype, device=output.device)
    if float(config.phase_noise_std_rad) > 0.0:
        steps = torch.randn_like(phase) * float(config.phase_noise_std_rad)
        phase = phase + torch.cumsum(steps, dim=-1)
    if float(config.carrier_frequency_offset_cycles) != 0.0:
        timeline = torch.arange(length, dtype=output.dtype, device=output.device)
        phase = phase + 2.0 * math.pi * float(
            config.carrier_frequency_offset_cycles
        ) * timeline[None, :] / float(length)
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    rotated_real = real * cosine - imag * sine
    rotated_imag = real * sine + imag * cosine
    output = torch.stack([rotated_real, rotated_imag], dim=1)

    if int(config.sample_shift):
        output = torch.roll(output, shifts=int(config.sample_shift), dims=-1)
    mask_length = int(round(length * float(config.local_mask_fraction)))
    if mask_length < 0 or mask_length > length:
        raise ValueError("local_mask_fraction must be in [0, 1]")
    if mask_length:
        start = (length - mask_length) // 2
        output[:, :, start : start + mask_length] = 0.0
    return output.squeeze(0) if squeezed else output


def receiver_robustness_grid():
    """A compact, preregisterable grid for T6 receiver-shift evaluation."""

    return {
        "clean": ReceiverPerturbationConfig(),
        "gain_1db": ReceiverPerturbationConfig(iq_gain_imbalance_db=1.0),
        "quadrature_3deg": ReceiverPerturbationConfig(quadrature_error_deg=3.0),
        "phase_noise_1e-3": ReceiverPerturbationConfig(
            phase_noise_std_rad=1e-3
        ),
        "cfo_0p01": ReceiverPerturbationConfig(
            carrier_frequency_offset_cycles=0.01
        ),
        "combined_mild": ReceiverPerturbationConfig(
            iq_gain_imbalance_db=0.5,
            quadrature_error_deg=2.0,
            phase_noise_std_rad=5e-4,
            carrier_frequency_offset_cycles=0.005,
        ),
    }
