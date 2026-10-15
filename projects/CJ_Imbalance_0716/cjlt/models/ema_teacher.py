"""Exponential-moving-average teacher without gradient updates."""

import copy
import math

import torch
from torch import nn


def ema_momentum(progress, start=0.99, end=0.999):
    value = min(1.0, max(0.0, float(progress)))
    cosine = 0.5 * (1.0 - math.cos(math.pi * value))
    return float(start) + (float(end) - float(start)) * cosine


class EMATeacher(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student, progress):
        momentum = ema_momentum(progress)
        student_state = student.state_dict()
        for name, teacher_value in self.model.state_dict().items():
            student_value = student_state[name].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(momentum).add_(student_value, alpha=1.0 - momentum)
            else:
                teacher_value.copy_(student_value)
        return momentum

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)

