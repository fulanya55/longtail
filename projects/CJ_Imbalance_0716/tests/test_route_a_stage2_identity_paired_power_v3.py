import numpy as np
import torch

from experiments.run_route_a_stage2_identity_paired_power_v3 import _pair_losses, _power_weak_mask


def test_power_weak_mask_requires_target_fixed_companion_and_low_level():
    labels=np.array([[1,0,0,1,0,0],[0,0,0,1,1,0],[1,0,0,0,1,0]],dtype=np.uint8)
    weak20=_power_weak_mask(labels,20,10)
    weak25=_power_weak_mask(labels,25,10)
    assert weak20[0,0]
    assert not weak20[1,0]
    assert weak20[2,1]
    assert not weak25.any()


def test_pair_losses_are_finite_and_masked():
    student_logits=torch.randn(4,3,requires_grad=True)
    teacher_logits=torch.randn(4,3)
    student_maps=torch.randn(4,3,10,requires_grad=True)
    teacher_maps=torch.randn(4,3,10)
    mask=torch.tensor([[1,0,1],[0,1,0],[1,1,0],[0,0,1]],dtype=torch.bool)
    score,map_loss=_pair_losses(student_logits,student_maps,teacher_logits,teacher_maps,mask)
    (score+map_loss).backward()
    assert torch.isfinite(score) and torch.isfinite(map_loss)
    assert torch.isfinite(student_logits.grad).all()

