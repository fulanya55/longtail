"""Minimal complex-valued layers used by the validated I/Q interface."""

import torch
from torch import nn


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.real_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.imag_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_channels))
            self.imag_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("real_bias", None)
            self.register_parameter("imag_bias", None)

    def forward(self, real, imag):
        output_real = self.real_conv(real) - self.imag_conv(imag)
        output_imag = self.real_conv(imag) + self.imag_conv(real)
        if self.real_bias is not None:
            output_real = output_real + self.real_bias.view(1, -1, 1)
            output_imag = output_imag + self.imag_bias.view(1, -1, 1)
        return output_real, output_imag


def complex_magnitude(real, imag, epsilon=1e-8):
    return torch.sqrt(real.pow(2) + imag.pow(2) + float(epsilon))
