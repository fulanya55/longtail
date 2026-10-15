"""Complex-domain weak/strong views with parameter-safety metadata."""

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class AugmentedComplexView:
    signal: torch.Tensor
    preserves_parameter_mask: torch.Tensor
    parameter_transform: str
    classification_only: bool


def _batched(signal):
    values = torch.as_tensor(signal).float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
        squeezed = True
    elif values.ndim == 3:
        squeezed = False
    else:
        raise ValueError("signal must have shape [2, L] or [B, 2, L]")
    if values.shape[1] != 2:
        raise ValueError("complex signal must use real/imag channels")
    return values.clone(), squeezed


def _uniform(shape, low, high, device, generator):
    return low + (high - low) * torch.rand(
        shape, device=device, generator=generator
    )


def _phase_rotate(values, maximum, generator):
    angle = _uniform((len(values), 1), -maximum, maximum, values.device, generator)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    real = values[:, 0] * cosine - values[:, 1] * sine
    imag = values[:, 0] * sine + values[:, 1] * cosine
    return torch.stack([real, imag], dim=1)


def _roll_each(values, minimum_fraction, maximum_fraction, generator):
    length = values.shape[-1]
    output = values.clone()
    magnitudes = _uniform(
        (len(values),), minimum_fraction, maximum_fraction, values.device, generator
    )
    signs = torch.where(
        torch.rand(len(values), device=values.device, generator=generator) > 0.5,
        torch.ones(len(values), device=values.device),
        -torch.ones(len(values), device=values.device),
    )
    shifts = torch.round(magnitudes * signs * float(length)).long()
    for index, shift in enumerate(shifts.tolist()):
        output[index] = torch.roll(output[index], shifts=int(shift), dims=-1)
    return output


def _add_noise(values, max_db, generator, min_db=0.0):
    level_db = _uniform(
        (len(values), 1, 1), float(min_db), float(max_db), values.device, generator
    )
    rms = torch.sqrt((values.pow(2).sum(dim=1, keepdim=True)).mean(dim=-1, keepdim=True).clamp_min(1e-12))
    noise_scale = rms * torch.pow(torch.tensor(10.0, device=values.device), -level_db / 20.0)
    noise = torch.randn(values.shape, device=values.device, generator=generator)
    return values + noise * noise_scale


def _parameter_mask(batch, parameter_names: Sequence[str], preserves_jnr):
    mask = torch.ones(batch, len(parameter_names), dtype=torch.bool)
    for index, name in enumerate(parameter_names):
        if "jnr" in str(name).lower() and not preserves_jnr:
            mask[:, index] = False
    return mask


def weak_complex_augmentation(signal, parameter_names, generator=None):
    values, squeezed = _batched(signal)
    values = _phase_rotate(values, maximum=0.20, generator=generator)
    scale = _uniform((len(values), 1, 1), 0.95, 1.05, values.device, generator)
    values = values * scale
    values = _roll_each(values, 0.0, 0.005, generator)
    values = _add_noise(
        values, min_db=18.0, max_db=30.0, generator=generator
    )
    mask = _parameter_mask(len(values), parameter_names, preserves_jnr=False)
    if squeezed:
        values = values.squeeze(0)
        mask = mask.squeeze(0)
    return AugmentedComplexView(
        signal=values,
        preserves_parameter_mask=mask,
        parameter_transform="identity_except_jnr",
        classification_only=False,
    )


def strong_complex_augmentation(signal, parameter_names, generator=None):
    values, squeezed = _batched(signal)
    values = _phase_rotate(values, maximum=3.14159265, generator=generator)
    scale = _uniform((len(values), 1, 1), 0.8, 1.2, values.device, generator)
    values = values * scale
    values = _roll_each(values, 0.01, 0.02, generator)
    gain_error = _uniform((len(values), 1), -0.08, 0.08, values.device, generator)
    real = values[:, 0] * (1.0 + gain_error)
    imag = values[:, 1] * (1.0 - gain_error)
    quadrature_leak = _uniform((len(values), 1), -0.05, 0.05, values.device, generator)
    imag = imag + quadrature_leak * real
    values = torch.stack([real, imag], dim=1)
    phase_steps = 0.002 * torch.randn(
        (len(values), values.shape[-1]),
        device=values.device,
        generator=generator,
    )
    phase_noise = torch.cumsum(phase_steps, dim=-1)
    cosine = torch.cos(phase_noise)
    sine = torch.sin(phase_noise)
    real = values[:, 0] * cosine - values[:, 1] * sine
    imag = values[:, 0] * sine + values[:, 1] * cosine
    values = torch.stack([real, imag], dim=1)
    values = _add_noise(values, max_db=6.0, generator=generator)
    max_mask = max(1, int(round(values.shape[-1] * 0.03)))
    for index in range(len(values)):
        length = int(
            torch.randint(0, max_mask + 1, (1,), generator=generator).item()
        )
        if length:
            start = int(
                torch.randint(
                    0, values.shape[-1] - length + 1, (1,), generator=generator
                ).item()
            )
            values[index, :, start : start + length] = 0.0
    mask = _parameter_mask(len(values), parameter_names, preserves_jnr=False)
    if squeezed:
        values = values.squeeze(0)
        mask = mask.squeeze(0)
    return AugmentedComplexView(
        signal=values,
        preserves_parameter_mask=mask,
        parameter_transform="identity_except_jnr",
        classification_only=False,
    )
