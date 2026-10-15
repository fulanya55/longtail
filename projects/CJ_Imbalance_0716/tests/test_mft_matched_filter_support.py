import torch
from cjlt.models.mft_matched_filter_support import effective_delay_support, fixed_phase_permutation, lfm_reference


def test_support_is_finite_and_higher_for_eight_separated_lfm_copies():
    ref = lfm_reference(); length = 3000
    def record(count):
        z = torch.zeros(length, dtype=torch.complex64)
        for delay in torch.linspace(420, 960, count).round().long(): z += torch.roll(torch.nn.functional.pad(ref, (0, length-ref.numel())), int(delay))
        return torch.stack((z.real, z.imag))[None]
    three, eight = record(3), record(8)
    score3 = effective_delay_support(three, ref); score8 = effective_delay_support(eight, ref)
    control = effective_delay_support(eight, fixed_phase_permutation(ref))
    assert torch.isfinite(torch.cat((score3, score8, control))).all()
    assert score8.item() > score3.item()
