"""Fixed three-atom complex OMP residual for paired MFT physics screens."""
from __future__ import annotations

import torch

from cjlt.models.mft_matched_filter_support import fixed_phase_permutation, lfm_reference


def three_atom_residual_fraction(signal_iq: torch.Tensor, reference: torch.Tensor,
                                delay_start: int = 400, delay_stop: int = 1000,
                                atoms: int = 3) -> torch.Tensor:
    """Energy fraction unexplained by a fixed complex three-atom OMP model."""
    if signal_iq.ndim != 3 or signal_iq.shape[1] != 2:
        raise ValueError('signal_iq must have shape [batch, 2, time]')
    if reference.ndim != 1 or reference.numel() < 2:
        raise ValueError('reference must be a one-dimensional LFM')
    length = int(signal_iq.shape[-1]); start, stop = int(delay_start), int(delay_stop)
    if not (0 <= start < stop <= length) or int(atoms) < 1:
        raise ValueError('invalid OMP dictionary contract')
    z = torch.complex(signal_iq[:, 0].float(), signal_iq[:, 1].float())
    template = torch.zeros(length, device=z.device, dtype=torch.complex64)
    template[:reference.numel()] = reference.to(device=z.device, dtype=torch.complex64)
    dictionary = torch.stack([torch.roll(template, int(delay)) for delay in range(start, stop)], dim=0)
    batch = int(z.shape[0]); chosen = torch.zeros(batch, 0, dtype=torch.long, device=z.device)
    used = torch.zeros(batch, dictionary.shape[0], dtype=torch.bool, device=z.device)
    residual = z
    for _ in range(int(atoms)):
        correlations = torch.abs(residual @ dictionary.conj().transpose(0, 1)).masked_fill(used, float('-inf'))
        selected = correlations.argmax(dim=1); used.scatter_(1, selected[:, None], True)
        chosen = torch.cat((chosen, selected[:, None]), dim=1); selected_atoms = dictionary[chosen]
        gram = torch.einsum('bkt,blt->bkl', selected_atoms.conj(), selected_atoms)
        rhs = torch.einsum('bkt,bt->bk', selected_atoms.conj(), z)
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=z.device)[None]
        coefficients = torch.linalg.solve(gram + eye * 1.0e-5, rhs[..., None]).squeeze(-1)
        residual = z - torch.einsum('bkt,bk->bt', selected_atoms, coefficients)
    return residual.abs().square().sum(dim=1) / z.abs().square().sum(dim=1).clamp_min(1.0e-12)


__all__ = ['fixed_phase_permutation', 'lfm_reference', 'three_atom_residual_fraction']
