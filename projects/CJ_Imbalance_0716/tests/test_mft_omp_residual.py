import torch

from cjlt.models.mft_omp_residual import lfm_reference, three_atom_residual_fraction


def test_three_atom_residual_is_higher_for_eight_than_three_noiseless_copies():
    ref = lfm_reference(); length = 3000; template = torch.nn.functional.pad(ref, (0, length - ref.numel()))
    def record(count):
        z = torch.zeros(length, dtype=torch.complex64)
        for delay in torch.linspace(420, 960, count).round().long(): z += torch.roll(template, int(delay))
        return torch.stack((z.real, z.imag))[None]
    score3 = three_atom_residual_fraction(record(3), ref)
    score8 = three_atom_residual_fraction(record(8), ref)
    assert torch.isfinite(torch.cat((score3, score8))).all()
    assert score8.item() > score3.item()
