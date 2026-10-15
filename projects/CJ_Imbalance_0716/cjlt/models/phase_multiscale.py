"""Phase-aware multi-scale complex I/Q feature extractor.

The encoder keeps real/imaginary streams complex-valued beyond the first stem,
adds stable phase-difference descriptors, and uses sparse temporal attention at
different semantic depths.  It follows the same public backbone contract as
``LightweightComplexIQClassifier`` so it can be used by the existing TAPC
training and evaluation pipeline.
"""

import torch
from torch import nn
from torch.nn import functional as F

from cjlt.models.complex_layers import ComplexConv1d, complex_magnitude


class ComplexBatchNormActivation1d(nn.Module):
    def __init__(self, channels, activation=True):
        super().__init__()
        self.real_bn = nn.BatchNorm1d(int(channels))
        self.imag_bn = nn.BatchNorm1d(int(channels))
        self.activation = bool(activation)

    def forward(self, real, imag):
        real = self.real_bn(real)
        imag = self.imag_bn(imag)
        if self.activation:
            real = F.silu(real, inplace=True)
            imag = F.silu(imag, inplace=True)
        return real, imag


class ComplexResidualBlock1d(nn.Module):
    """Two complex convolutions with a complex residual projection."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv1 = ComplexConv1d(
            int(in_channels), int(out_channels), int(kernel_size), padding=padding
        )
        self.norm1 = ComplexBatchNormActivation1d(out_channels, activation=True)
        self.conv2 = ComplexConv1d(
            int(out_channels), int(out_channels), int(kernel_size), padding=padding
        )
        self.norm2 = ComplexBatchNormActivation1d(out_channels, activation=False)
        self.shortcut = None
        self.shortcut_norm = None
        if int(in_channels) != int(out_channels):
            self.shortcut = ComplexConv1d(
                int(in_channels), int(out_channels), kernel_size=1, padding=0
            )
            self.shortcut_norm = ComplexBatchNormActivation1d(
                out_channels, activation=False
            )

    def forward(self, real, imag):
        residual_real, residual_imag = real, imag
        real, imag = self.conv1(real, imag)
        real, imag = self.norm1(real, imag)
        real, imag = self.conv2(real, imag)
        real, imag = self.norm2(real, imag)
        if self.shortcut is not None:
            residual_real, residual_imag = self.shortcut(
                residual_real, residual_imag
            )
            residual_real, residual_imag = self.shortcut_norm(
                residual_real, residual_imag
            )
        return (
            F.silu(real + residual_real, inplace=True),
            F.silu(imag + residual_imag, inplace=True),
        )


class ComplexResidualStem1d(nn.Module):
    """Preserve complex interactions through two residual feature levels."""

    def __init__(self, width):
        super().__init__()
        width = int(width)
        self.block1 = ComplexResidualBlock1d(1, width, kernel_size=7)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.block2 = ComplexResidualBlock1d(
            width, width * 2, kernel_size=5
        )
        self.out_channels = width * 2

    def forward(self, real, imag):
        real, imag = self.block1(real, imag)
        real = self.pool(real)
        imag = self.pool(imag)
        return self.block2(real, imag)


class FixedBlurPool1d(nn.Module):
    """Fixed binomial low-pass used immediately before stride-2 pooling.

    The module performs only the anti-alias prefilter.  The existing AvgPool1d
    remains responsible for decimation, so all registered stage lengths stay
    unchanged.  One buffer is expanded across channels; there are no trainable
    parameters and real/imaginary streams use exactly the same coefficients.
    """

    def __init__(self, coefficients=(1.0, 4.0, 6.0, 4.0, 1.0)):
        super().__init__()
        kernel = torch.as_tensor(coefficients, dtype=torch.float32)
        if kernel.ndim != 1 or kernel.numel() % 2 != 1 or kernel.numel() < 3:
            raise ValueError("blur coefficients must be a one-dimensional odd kernel")
        if not torch.isfinite(kernel).all() or float(kernel.sum()) <= 0.0:
            raise ValueError("blur coefficients must be finite with positive sum")
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.padding = int(kernel.numel() // 2)

    def forward(self, values):
        if values.ndim != 3:
            raise ValueError("FixedBlurPool1d expects [batch, channels, length]")
        if values.shape[-1] <= self.padding:
            raise ValueError("input is too short for reflected blur padding")
        weight = self.kernel.to(dtype=values.dtype).expand(values.shape[1], 1, -1)
        padded = F.pad(values, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, weight, groups=values.shape[1])


class CrossStageComplexFeaturePyramid1d(nn.Module):
    """Align Stage2/3/4 complex evidence and add a bounded residual.

    The pyramid reads information that already exists in the B0 encoder.  It
    uses fixed anti-alias filtering for temporal alignment and contains no
    attention or multiplicative feature gate.
    """

    def __init__(
        self,
        stage_channels=(32, 48, 64),
        projection_channels=64,
        output_channels=128,
        residual_bound=0.25,
    ):
        super().__init__()
        self.stage_channels = tuple(int(value) for value in stage_channels)
        projection_channels = int(projection_channels)
        output_channels = int(output_channels)
        if len(self.stage_channels) != 3:
            raise ValueError("CSFP requires exactly Stage2, Stage3, and Stage4")
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels * 3, projection_channels, 1, bias=False),
                    nn.GroupNorm(8, projection_channels),
                    nn.SiLU(inplace=True),
                )
                for channels in self.stage_channels
            ]
        )
        self.blur = FixedBlurPool1d()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fusion = nn.Sequential(
            nn.Conv1d(projection_channels * 3, output_channels, 1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.residual_bound = float(residual_bound)
        self.output_channels = output_channels

    @staticmethod
    def _complex_components(real, imag):
        return torch.cat([real, imag, complex_magnitude(real, imag)], dim=1)

    def _downsample(self, values, levels):
        for _ in range(int(levels)):
            values = self.pool(self.blur(values))
        return values

    def forward_pyramid(self, stage2, stage3, stage4):
        stage2_feature = self.projections[0](self._complex_components(*stage2))
        stage3_feature = self.projections[1](self._complex_components(*stage3))
        stage4_feature = self.projections[2](self._complex_components(*stage4))
        stage2_feature = self._downsample(stage2_feature, 2)
        stage3_feature = self._downsample(stage3_feature, 1)
        if not (
            stage2_feature.shape[-1]
            == stage3_feature.shape[-1]
            == stage4_feature.shape[-1]
        ):
            raise ValueError("CSFP stage lengths do not align")
        return self.fusion(
            torch.cat([stage2_feature, stage3_feature, stage4_feature], dim=1)
        )

    def forward(self, base_temporal, stage2, stage3, stage4):
        pyramid = self.forward_pyramid(stage2, stage3, stage4)
        if pyramid.shape != base_temporal.shape:
            raise ValueError("CSFP residual and B0 temporal feature shapes differ")
        scale = self.residual_bound * torch.tanh(self.residual_scale)
        return base_temporal + scale * pyramid


class ComplexMultiScaleResidualStage1d(nn.Module):
    """Downsample and aggregate short, medium, and long complex evidence."""

    def __init__(
        self,
        in_channels,
        out_channels,
        branch_channels,
        kernels=(3, 7, 15),
        dilations=(1, 2, 4),
        anti_aliased_downsampling=False,
    ):
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have the same length")
        self.pre_downsample = (
            FixedBlurPool1d() if anti_aliased_downsampling else nn.Identity()
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernels, dilations):
            kernel_size = int(kernel_size)
            dilation = int(dilation)
            self.branches.append(
                nn.ModuleDict(
                    {
                        "conv": ComplexConv1d(
                            int(in_channels),
                            int(branch_channels),
                            kernel_size=kernel_size,
                            padding=dilation * (kernel_size // 2),
                            dilation=dilation,
                            bias=False,
                        ),
                        "norm": ComplexBatchNormActivation1d(
                            int(branch_channels), activation=True
                        ),
                    }
                )
            )
        merged_channels = int(branch_channels) * len(self.branches)
        self.projection = ComplexConv1d(
            merged_channels, int(out_channels), kernel_size=1, bias=False
        )
        self.projection_norm = ComplexBatchNormActivation1d(
            int(out_channels), activation=False
        )
        self.shortcut = ComplexConv1d(
            int(in_channels), int(out_channels), kernel_size=1, bias=False
        )
        self.shortcut_norm = ComplexBatchNormActivation1d(
            int(out_channels), activation=False
        )

    def forward(self, real, imag):
        real = self.pre_downsample(real)
        imag = self.pre_downsample(imag)
        real = self.downsample(real)
        imag = self.downsample(imag)
        residual_real, residual_imag = self.shortcut(real, imag)
        residual_real, residual_imag = self.shortcut_norm(
            residual_real, residual_imag
        )
        branch_real = []
        branch_imag = []
        for branch in self.branches:
            output_real, output_imag = branch["conv"](real, imag)
            output_real, output_imag = branch["norm"](
                output_real, output_imag
            )
            branch_real.append(output_real)
            branch_imag.append(output_imag)
        encoded_real, encoded_imag = self.projection(
            torch.cat(branch_real, dim=1), torch.cat(branch_imag, dim=1)
        )
        encoded_real, encoded_imag = self.projection_norm(
            encoded_real, encoded_imag
        )
        return (
            F.silu(encoded_real + residual_real, inplace=True),
            F.silu(encoded_imag + residual_imag, inplace=True),
        )


class DelayedComplexMultiScaleEncoder1d(nn.Module):
    """Keep real/imaginary streams separate through all long-range stages."""

    def __init__(self, width, anti_aliased_downsampling=False):
        super().__init__()
        width = int(width)
        self.block1 = ComplexResidualBlock1d(1, width, kernel_size=7)
        self.pre_downsample = (
            FixedBlurPool1d() if anti_aliased_downsampling else nn.Identity()
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.block2 = ComplexResidualBlock1d(
            width, width * 2, kernel_size=5
        )
        self.stage3 = ComplexMultiScaleResidualStage1d(
            width * 2,
            width * 3,
            branch_channels=width,
            anti_aliased_downsampling=anti_aliased_downsampling,
        )
        self.stage4 = ComplexMultiScaleResidualStage1d(
            width * 3,
            width * 4,
            branch_channels=width + width // 2,
            anti_aliased_downsampling=anti_aliased_downsampling,
        )
        self.out_channels = width * 4

    def forward_pyramid(self, real, imag):
        """Return every complex stage, including the pre-downsample stage1."""
        stage1_real, stage1_imag = self.block1(real, imag)
        stage1_real = self.pre_downsample(stage1_real)
        stage1_imag = self.pre_downsample(stage1_imag)
        real = self.downsample(stage1_real)
        imag = self.downsample(stage1_imag)
        stage2_real, stage2_imag = self.block2(real, imag)
        stage3_real, stage3_imag = self.stage3(stage2_real, stage2_imag)
        stage4_real, stage4_imag = self.stage4(stage3_real, stage3_imag)
        return (
            (stage1_real, stage1_imag),
            (stage2_real, stage2_imag),
            (stage3_real, stage3_imag),
            (stage4_real, stage4_imag),
        )

    def forward_stages(self, real, imag):
        return self.forward_pyramid(real, imag)[1:]

    def forward(self, real, imag):
        return self.forward_stages(real, imag)[-1]


def _lagged(values, lag):
    lag = int(lag)
    if lag <= 0:
        raise ValueError("phase lag must be positive")
    if values.shape[-1] <= lag:
        return values[..., :1].expand_as(values)
    return torch.cat(
        [values[..., :1].expand(*values.shape[:-1], lag), values[..., :-lag]],
        dim=-1,
    )


class DelayedCartesianMagnitudePhaseFusion1d(nn.Module):
    """Late residual fusion without a single destructive concatenation bottleneck."""

    def __init__(
        self,
        complex_channels,
        output_channels,
        phase_hidden=32,
        phase_lags=(1, 4, 16),
        include_phase=True,
        initial_phase_scale=0.1,
    ):
        super().__init__()
        complex_channels = int(complex_channels)
        output_channels = int(output_channels)
        self.phase_lags = tuple(int(value) for value in phase_lags)
        self.include_phase = bool(include_phase)
        self.cartesian_projection = nn.Sequential(
            nn.Conv1d(
                complex_channels * 2, output_channels, 1, bias=False
            ),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(inplace=True),
        )
        self.magnitude_projection = nn.Sequential(
            nn.Conv1d(complex_channels, output_channels, 1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.SiLU(inplace=True),
        )
        if self.include_phase:
            phase_input_channels = (
                complex_channels * 2 * len(self.phase_lags)
            )
            self.phase_projection = nn.Sequential(
                nn.Conv1d(
                    phase_input_channels, int(phase_hidden), 1, bias=False
                ),
                nn.BatchNorm1d(int(phase_hidden)),
                nn.SiLU(inplace=True),
                nn.Conv1d(
                    int(phase_hidden), output_channels, 1, bias=False
                ),
                nn.BatchNorm1d(output_channels),
            )
            self.phase_scale = nn.Parameter(
                torch.tensor(float(initial_phase_scale))
            )
        else:
            self.phase_projection = None
            self.register_parameter("phase_scale", None)
        self.output_channels = output_channels

    def _phase_features(self, real, imag, magnitude):
        normalized_real = real / magnitude
        normalized_imag = imag / magnitude
        features = []
        for lag in self.phase_lags:
            previous_real = _lagged(normalized_real, lag)
            previous_imag = _lagged(normalized_imag, lag)
            features.extend(
                [
                    normalized_real * previous_real
                    + normalized_imag * previous_imag,
                    normalized_imag * previous_real
                    - normalized_real * previous_imag,
                ]
            )
        return torch.cat(features, dim=1)

    def forward_components(self, real, imag):
        """Return the auditable late-fusion branches before residual addition."""

        magnitude = complex_magnitude(real, imag)
        cartesian = self.cartesian_projection(torch.cat([real, imag], dim=1))
        stable_magnitude = self.magnitude_projection(magnitude)
        phase = None
        if self.phase_projection is not None:
            phase = self.phase_projection(
                self._phase_features(real, imag, magnitude)
            )
        return cartesian, stable_magnitude, phase

    def combine_components(self, cartesian, stable_magnitude, phase=None):
        fused = cartesian + stable_magnitude
        if phase is not None:
            fused = fused + torch.tanh(self.phase_scale) * phase
        return F.silu(fused, inplace=True)

    def forward(self, real, imag):
        return self.combine_components(*self.forward_components(real, imag))


class PhaseInteractionFusion1d(nn.Module):
    """Fuse complex components and normalized adjacent phase increments."""

    def __init__(self, complex_channels, out_channels, include_phase_delta=True):
        super().__init__()
        self.include_phase_delta = bool(include_phase_delta)
        input_multiplier = 5 if self.include_phase_delta else 3
        self.projection = nn.Sequential(
            nn.Conv1d(
                int(complex_channels) * input_multiplier,
                int(out_channels),
                1,
                bias=False,
            ),
            nn.BatchNorm1d(int(out_channels)),
            nn.SiLU(inplace=True),
        )

    def forward(self, real, imag):
        magnitude = complex_magnitude(real, imag)
        normalized_real = real / magnitude
        normalized_imag = imag / magnitude
        previous_real = torch.cat(
            [normalized_real[..., :1], normalized_real[..., :-1]], dim=-1
        )
        previous_imag = torch.cat(
            [normalized_imag[..., :1], normalized_imag[..., :-1]], dim=-1
        )
        phase_delta_cos = (
            normalized_real * previous_real
            + normalized_imag * previous_imag
        )
        phase_delta_sin = (
            normalized_imag * previous_real
            - normalized_real * previous_imag
        )
        components = [real, imag, magnitude]
        if self.include_phase_delta:
            components.extend([phase_delta_cos, phase_delta_sin])
        return self.projection(torch.cat(components, dim=1))


class ECA1d(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(
            1, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, values):
        weights = F.adaptive_avg_pool1d(values, 1)
        weights = weights.squeeze(-1).unsqueeze(1)
        weights = self.conv(weights).squeeze(1).unsqueeze(-1)
        return values * torch.sigmoid(weights)


class PCAF1d(nn.Module):
    def __init__(self, channels, kernel_size=5, dilations=(1, 3, 5), reduction=8):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=int(dilation) * (kernel_size // 2),
                    dilation=int(dilation),
                    groups=channels,
                    bias=False,
                )
                for dilation in dilations
            ]
        )
        self.mix = nn.Conv1d(channels, channels, 1, bias=False)
        hidden = max(8, channels // int(reduction))
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, values):
        encoded = sum(branch(values) for branch in self.depthwise)
        encoded = self.mix(encoded / float(len(self.depthwise)))
        weights = self.gate(F.adaptive_avg_pool1d(encoded, 1))
        return values * weights


class GlobalContext1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        channels = int(channels)
        hidden = max(8, channels // int(reduction))
        self.mask = nn.Conv1d(channels, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )

    def forward(self, values):
        attention = torch.softmax(self.mask(values), dim=-1)
        context = torch.sum(values * attention, dim=-1, keepdim=True)
        return values + self.transform(context)


class ResidualAttention1d(nn.Module):
    """Keep an explicit identity path around a feature-gating module."""

    def __init__(self, attention, initial_scale=0.1):
        super().__init__()
        self.attention = attention
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, values):
        attended = self.attention(values)
        return values + self.scale * (attended - values)


class MultiScaleDepthwiseResidual1d(nn.Module):
    """Parameter-efficient local, medium, and long temporal evidence block."""

    def __init__(
        self,
        channels,
        kernels=(3, 7, 15),
        dilations=(1, 2, 4),
        dropout=0.0,
    ):
        super().__init__()
        channels = int(channels)
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have the same length")
        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernels, dilations):
            kernel_size = int(kernel_size)
            dilation = int(dilation)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * len(self.branches), channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout1d(float(dropout))

    def forward(self, values):
        encoded = torch.cat([branch(values) for branch in self.branches], dim=1)
        encoded = self.dropout(self.fusion(encoded))
        return F.silu(values + encoded, inplace=True)


class EvidencePreservingMultiScaleBlock1d(nn.Module):
    """Ungated short/medium/long depthwise evidence residual."""

    def __init__(
        self,
        channels=128,
        kernels=(3, 15, 31),
        dilations=(1, 2, 4),
    ):
        super().__init__()
        channels = int(channels)
        if len(kernels) != 3 or len(dilations) != 3:
            raise ValueError("EPB requires exactly three preregistered scales")
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    int(kernel),
                    padding=int(dilation) * (int(kernel) // 2),
                    dilation=int(dilation),
                    groups=channels,
                    bias=False,
                )
                for kernel, dilation in zip(kernels, dilations)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values):
        encoded = self.fusion(
            torch.cat([branch(values) for branch in self.branches], dim=1)
        )
        return values + encoded


class TemporalStage1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, attention):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=3,
                stride=int(stride),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(int(out_channels)),
            nn.SiLU(inplace=True),
        )
        self.multiscale = MultiScaleDepthwiseResidual1d(out_channels)
        self.attention = attention

    def forward(self, values):
        values = self.multiscale(self.projection(values))
        return self.attention(values)


class AttentiveStatisticsPooling1d(nn.Module):
    """Retain both occurrence strength and temporal variation before projection."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        hidden = max(8, channels // 4)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, 1),
        )
        self.output_dim = channels * 2

    def forward(self, values):
        weights = torch.softmax(self.attention(values), dim=-1)
        mean = torch.sum(values * weights, dim=-1)
        second_moment = torch.sum(values.pow(2) * weights, dim=-1)
        standard_deviation = torch.sqrt(
            torch.clamp(second_moment - mean.pow(2), min=1e-5)
        )
        return torch.cat([mean, standard_deviation], dim=1)


class GlobalStatisticsPooling1d(nn.Module):
    """Unweighted mean/std path that attention cannot suppress."""

    def __init__(self, channels):
        super().__init__()
        self.output_dim = int(channels) * 2

    def forward(self, values):
        mean = values.mean(dim=-1)
        standard_deviation = torch.sqrt(
            torch.clamp(values.pow(2).mean(dim=-1) - mean.pow(2), min=1e-5)
        )
        return torch.cat([mean, standard_deviation], dim=1)


def _statistics_projection(input_dim, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(int(input_dim), int(output_dim)),
        nn.LayerNorm(int(output_dim)),
        nn.GELU(),
        nn.Dropout(float(dropout)),
    )


class DualPathStatisticsPooling1d(nn.Module):
    """Fuse a global safety path with the original attentive statistics path."""

    def __init__(self, channels, output_dim, dropout, initial_scale=0.1):
        super().__init__()
        channels = int(channels)
        output_dim = int(output_dim)
        self.global_pool = GlobalStatisticsPooling1d(channels)
        self.attentive_pool = AttentiveStatisticsPooling1d(channels)
        self.global_projection = _statistics_projection(
            self.global_pool.output_dim, output_dim, dropout
        )
        self.attentive_projection = _statistics_projection(
            self.attentive_pool.output_dim, output_dim, dropout
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.output_dim = output_dim

    def forward(self, values):
        global_features = self.global_projection(self.global_pool(values))
        attentive_features = self.attentive_projection(
            self.attentive_pool(values)
        )
        return global_features + torch.tanh(self.residual_scale) * attentive_features


class LocalEvidenceStatisticsBypass1d(nn.Module):
    """Bounded max/top-k evidence path beside the global mean/std vector."""

    def __init__(self, channels, output_dim, topk, dropout, residual_bound=0.25):
        super().__init__()
        channels = int(channels)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("local evidence top-k must be positive")
        self.local_projection = _statistics_projection(
            channels * 2, int(output_dim), dropout
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        self.residual_bound = float(residual_bound)

    def forward_local(self, temporal):
        if temporal.ndim != 3 or temporal.shape[-1] < self.topk:
            raise ValueError("temporal feature map is too short for local top-k")
        maximum = temporal.amax(dim=-1)
        topk_mean = torch.topk(
            temporal, k=self.topk, dim=-1, largest=True, sorted=False
        ).values.mean(dim=-1)
        return self.local_projection(torch.cat([maximum, topk_mean], dim=1))

    def forward(self, global_features, temporal):
        local_features = self.forward_local(temporal)
        scale = self.residual_bound * torch.tanh(self.residual_scale)
        return global_features + scale * local_features


class LabelQueryPooling1d(nn.Module):
    """Use one temporal query per label while retaining a uniform attention floor."""

    def __init__(self, channels, num_labels, query_dim=32, attention_floor=0.2):
        super().__init__()
        channels = int(channels)
        self.num_labels = int(num_labels)
        self.query_dim = int(query_dim)
        floor = float(attention_floor)
        if floor < 0.0 or floor >= 1.0:
            raise ValueError("attention_floor must be in [0, 1)")
        self.attention_floor = floor
        self.key_projection = nn.Conv1d(channels, self.query_dim, 1, bias=False)
        self.value_projection = nn.Conv1d(channels, self.query_dim, 1, bias=False)
        self.queries = nn.Parameter(torch.empty(self.num_labels, self.query_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.normalization = nn.LayerNorm(self.query_dim)

    def forward(self, values):
        keys = self.key_projection(values)
        value = self.value_projection(values)
        scores = torch.einsum("bdt,kd->bkt", keys, self.queries)
        scores = scores / float(self.query_dim) ** 0.5
        attention = torch.softmax(scores, dim=-1)
        if self.attention_floor:
            uniform = 1.0 / float(values.shape[-1])
            attention = (
                self.attention_floor * uniform
                + (1.0 - self.attention_floor) * attention
            )
        pooled = torch.einsum("bkt,bdt->bkd", attention, value)
        return self.normalization(pooled)


class PositionalLabelQueryPooling1d(LabelQueryPooling1d):
    """Label queries with deterministic absolute temporal position keys.

    Unlike ``LabelQueryPooling1d``, equal local content at different record
    positions is not indistinguishable to the query.  The sinusoidal code has
    no fitted data-dependent statistic and does not alter the value path.
    """

    def _position(self, length, device, dtype):
        position = torch.arange(int(length), device=device, dtype=dtype)
        index = torch.arange(self.query_dim, device=device, dtype=dtype)
        scale = torch.pow(10000.0, (2.0 * torch.floor(index / 2.0)) / float(self.query_dim))
        angles = position[:, None] / scale[None, :]
        code = torch.empty_like(angles)
        code[:, 0::2] = torch.sin(angles[:, 0::2])
        code[:, 1::2] = torch.cos(angles[:, 1::2])
        return code.transpose(0, 1).unsqueeze(0)

    def forward(self, values):
        keys = self.key_projection(values) + self._position(
            values.shape[-1], values.device, values.dtype
        )
        value = self.value_projection(values)
        scores = torch.einsum("bdt,kd->bkt", keys, self.queries)
        scores = scores / float(self.query_dim) ** 0.5
        attention = torch.softmax(scores, dim=-1)
        if self.attention_floor:
            attention = self.attention_floor / float(values.shape[-1]) + (1.0 - self.attention_floor) * attention
        return self.normalization(torch.einsum("bkt,bdt->bkd", attention, value))


class DualPathLabelQueryPooling1d(nn.Module):
    """Return a global safety vector and flattened class-specific query vectors."""

    def __init__(
        self,
        channels,
        num_labels,
        global_dim,
        query_dim,
        dropout,
        attention_floor,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.query_dim = int(query_dim)
        self.global_pool = GlobalStatisticsPooling1d(channels)
        self.global_projection = _statistics_projection(
            self.global_pool.output_dim, global_dim, dropout
        )
        self.query_pool = LabelQueryPooling1d(
            channels,
            num_labels=self.num_labels,
            query_dim=self.query_dim,
            attention_floor=attention_floor,
        )
        self.global_dim = int(global_dim)
        self.output_dim = self.global_dim + self.num_labels * self.query_dim

    def forward(self, values):
        global_features = self.global_projection(self.global_pool(values))
        query_features = self.query_pool(values).flatten(start_dim=1)
        return torch.cat([global_features, query_features], dim=1)


class DualPathPositionalLabelQueryPooling1d(DualPathLabelQueryPooling1d):
    """Same global safety path and feature width, with positional queries."""

    def __init__(self, channels, num_labels, global_dim, query_dim, dropout, attention_floor):
        super().__init__(channels, num_labels, global_dim, query_dim, dropout, attention_floor)
        self.query_pool = PositionalLabelQueryPooling1d(
            channels,
            num_labels=self.num_labels, query_dim=self.query_dim,
            attention_floor=attention_floor,
        )


class GlobalResidualLabelQueryClassifier(nn.Module):
    """Safe global logits plus a small class-specific query residual."""

    def __init__(
        self,
        global_dim,
        num_labels,
        query_dim,
        dropout,
        initial_scale=0.1,
    ):
        super().__init__()
        self.global_dim = int(global_dim)
        self.num_labels = int(num_labels)
        self.query_dim = int(query_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.global_classifier = nn.Linear(self.global_dim, self.num_labels)
        self.query_weight = nn.Parameter(
            torch.empty(self.num_labels, self.query_dim)
        )
        self.query_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.residual_scale = nn.Parameter(
            torch.full((self.num_labels,), float(initial_scale))
        )
        nn.init.xavier_uniform_(self.query_weight)

    def forward(self, features):
        expected = self.global_dim + self.num_labels * self.query_dim
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError(
                "label-query features must have shape [B, {}], got {}".format(
                    expected, tuple(features.shape)
                )
            )
        global_features = self.dropout(features[:, : self.global_dim])
        query_features = features[:, self.global_dim :].reshape(
            -1, self.num_labels, self.query_dim
        )
        query_features = self.dropout(query_features)
        global_logits = self.global_classifier(global_features)
        query_logits = torch.einsum(
            "bkd,kd->bk", query_features, self.query_weight
        ) + self.query_bias[None, :]
        return global_logits + torch.tanh(self.residual_scale)[None, :] * query_logits


class GlobalAveragePooling1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.output_dim = int(channels)

    def forward(self, values):
        return F.adaptive_avg_pool1d(values, 1).squeeze(-1)


class PhaseAwareMultiScaleIQClassifier(nn.Module):
    """Complex residual stem plus phase-aware multi-scale temporal encoder."""

    def __init__(
        self,
        num_classes=6,
        width=16,
        dropout=0.2,
        include_phase_delta=True,
        use_sparse_attention=True,
        attentive_statistics_pooling=True,
        pooling_mode="single",
        residual_attention=False,
        query_dim=32,
        attention_floor=0.2,
    ):
        super().__init__()
        width = int(width)
        if width < 4:
            raise ValueError("width must be at least 4")
        self.input_mode = "complex_1d_true"
        self.num_classes = int(num_classes)
        self.width = width
        normalized_pooling = str(pooling_mode).strip().lower()
        if normalized_pooling not in (
            "single",
            "dual_pool",
            "label_query",
            "positional_label_query",
        ):
            raise ValueError(
                "pooling_mode must be single, dual_pool, label_query, or "
                "positional_label_query"
            )
        self.feature_dim = width * 8
        self.architecture_options = {
            "include_phase_delta": bool(include_phase_delta),
            "use_sparse_attention": bool(use_sparse_attention),
            "attentive_statistics_pooling": bool(
                attentive_statistics_pooling
            ),
            "pooling_mode": normalized_pooling,
            "residual_attention": bool(residual_attention),
            "query_dim": int(query_dim),
            "attention_floor": float(attention_floor),
        }
        self.complex_stem = ComplexResidualStem1d(width)
        # Compatibility handles used by the existing freeze/eval helpers.
        self.complex_real_bn = nn.Identity()
        self.complex_imag_bn = nn.Identity()
        fusion_channels = width * 4
        self.phase_fusion = PhaseInteractionFusion1d(
            self.complex_stem.out_channels,
            fusion_channels,
            include_phase_delta=include_phase_delta,
        )
        stage1_attention = (
            ECA1d(fusion_channels, kernel_size=5)
            if use_sparse_attention
            else nn.Identity()
        )
        stage2_attention = (
            PCAF1d(width * 6) if use_sparse_attention else nn.Identity()
        )
        stage3_attention = (
            GlobalContext1d(width * 8)
            if use_sparse_attention
            else nn.Identity()
        )
        if use_sparse_attention and residual_attention:
            stage1_attention = ResidualAttention1d(stage1_attention)
            stage2_attention = ResidualAttention1d(stage2_attention)
            stage3_attention = ResidualAttention1d(stage3_attention)
        self.temporal_backbone = nn.Sequential(
            TemporalStage1d(
                fusion_channels,
                fusion_channels,
                stride=2,
                attention=stage1_attention,
            ),
            TemporalStage1d(
                fusion_channels,
                width * 6,
                stride=2,
                attention=stage2_attention,
            ),
            TemporalStage1d(
                width * 6,
                width * 8,
                stride=1,
                attention=stage3_attention,
            ),
        )
        if normalized_pooling == "dual_pool":
            self.global_pool = DualPathStatisticsPooling1d(
                width * 8,
                output_dim=self.feature_dim,
                dropout=dropout,
            )
            self.feature_projection = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(self.feature_dim, self.num_classes),
            )
        elif normalized_pooling in ("label_query", "positional_label_query"):
            global_dim = width * 8
            query_pool_class = (
                DualPathPositionalLabelQueryPooling1d
                if normalized_pooling == "positional_label_query"
                else DualPathLabelQueryPooling1d
            )
            self.global_pool = query_pool_class(
                width * 8,
                num_labels=self.num_classes,
                global_dim=global_dim,
                query_dim=int(query_dim),
                dropout=dropout,
                attention_floor=attention_floor,
            )
            self.feature_dim = self.global_pool.output_dim
            self.feature_projection = nn.Identity()
            self.classifier = GlobalResidualLabelQueryClassifier(
                global_dim=global_dim,
                num_labels=self.num_classes,
                query_dim=int(query_dim),
                dropout=dropout,
            )
        else:
            if attentive_statistics_pooling:
                self.global_pool = AttentiveStatisticsPooling1d(width * 8)
            else:
                self.global_pool = GlobalAveragePooling1d(width * 8)
            self.feature_projection = nn.Sequential(
                nn.Linear(self.global_pool.output_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(self.feature_dim, self.num_classes),
            )

    @staticmethod
    def _validate_input(values):
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(
                "complex_1d_true expects [batch, 2, length], got {}".format(
                    tuple(values.shape)
                )
            )

    def forward_temporal_features(self, values):
        self._validate_input(values)
        real = values[:, 0:1, :]
        imag = values[:, 1:2, :]
        real, imag = self.complex_stem(real, imag)
        fused = self.phase_fusion(real, imag)
        return self.temporal_backbone(fused)

    def forward_features(self, values):
        temporal = self.forward_temporal_features(values)
        return self.feature_projection(self.global_pool(temporal))

    def forward_classifier(self, features):
        return self.classifier(features)

    def forward(self, values, return_features=False):
        features = self.forward_features(values)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits


class DelayedFusionMultiScaleIQClassifier(nn.Module):
    """Deep complex multi-scale encoder with late residual I/Q fusion."""

    def __init__(
        self,
        num_classes=6,
        width=16,
        dropout=0.2,
        include_phase=True,
        phase_lags=(1, 4, 16),
        anti_aliased_downsampling=False,
        cross_stage_pyramid=False,
        evidence_preserving_block=False,
        local_statistics_topk=None,
    ):
        super().__init__()
        width = int(width)
        if width < 4:
            raise ValueError("width must be at least 4")
        self.input_mode = "complex_1d_true"
        self.num_classes = int(num_classes)
        self.width = width
        self.feature_dim = width * 8
        self.architecture_options = {
            "delayed_fusion": True,
            "complex_multiscale": True,
            "include_phase": bool(include_phase),
            "phase_lags": tuple(int(value) for value in phase_lags),
            "anti_aliased_downsampling": bool(anti_aliased_downsampling),
            "cross_stage_pyramid": bool(cross_stage_pyramid),
            "evidence_preserving_block": bool(evidence_preserving_block),
            "local_statistics_topk": (
                None if local_statistics_topk is None else int(local_statistics_topk)
            ),
            "attention": False,
            "pooling": "global_statistics",
        }
        self.complex_encoder = DelayedComplexMultiScaleEncoder1d(
            width,
            anti_aliased_downsampling=anti_aliased_downsampling,
        )
        self.complex_stem = self.complex_encoder
        self.complex_real_bn = nn.Identity()
        self.complex_imag_bn = nn.Identity()
        self.late_fusion = DelayedCartesianMagnitudePhaseFusion1d(
            self.complex_encoder.out_channels,
            output_channels=self.feature_dim,
            phase_hidden=width * 2,
            phase_lags=phase_lags,
            include_phase=include_phase,
            initial_phase_scale=0.1,
        )
        self.temporal_backbone = nn.Sequential(
            MultiScaleDepthwiseResidual1d(
                self.feature_dim, dropout=dropout
            )
        )
        self.global_pool = GlobalStatisticsPooling1d(self.feature_dim)
        self.feature_projection = _statistics_projection(
            self.global_pool.output_dim, self.feature_dim, dropout
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, self.num_classes),
        )
        # Instantiate the candidate-only module after every B0 module so a
        # shared torch seed produces byte-identical initialization for the
        # complete B0 parameter subset.
        self.cross_stage_pyramid = (
            CrossStageComplexFeaturePyramid1d(
                stage_channels=(width * 2, width * 3, width * 4),
                projection_channels=width * 4,
                output_channels=self.feature_dim,
                residual_bound=0.25,
            )
            if cross_stage_pyramid
            else None
        )
        self.evidence_preserving_block = (
            EvidencePreservingMultiScaleBlock1d(self.feature_dim)
            if evidence_preserving_block
            else None
        )
        self.local_statistics_bypass = (
            LocalEvidenceStatisticsBypass1d(
                self.feature_dim,
                self.feature_dim,
                topk=int(local_statistics_topk),
                dropout=dropout,
                residual_bound=0.25,
            )
            if local_statistics_topk is not None
            else None
        )

    def _pool_features(self, temporal):
        global_features = self.feature_projection(self.global_pool(temporal))
        if self.local_statistics_bypass is not None:
            return self.local_statistics_bypass(global_features, temporal)
        return global_features

    @staticmethod
    def _validate_input(values):
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(
                "complex_1d_true expects [batch, 2, length], got {}".format(
                    tuple(values.shape)
                )
            )
        if values.shape[-1] < 8:
            raise ValueError("DFMS-IQ requires an input length of at least 8")

    def forward_complex_features(self, values):
        self._validate_input(values)
        return self.complex_encoder(values[:, 0:1, :], values[:, 1:2, :])

    def forward_structure_maps(self, values):
        """Expose frozen global, temporal, and early complex maps for add-ons."""

        self._validate_input(values)
        stage1, stage2, stage3, stage4 = self.complex_encoder.forward_pyramid(
            values[:, 0:1, :], values[:, 1:2, :]
        )
        late_fusion = self.late_fusion(*stage4)
        pyramid_residual = None
        if self.cross_stage_pyramid is not None:
            pyramid_residual = self.cross_stage_pyramid.forward_pyramid(
                stage2, stage3, stage4
            )
            late_fusion = late_fusion + (
                self.cross_stage_pyramid.residual_bound
                * torch.tanh(self.cross_stage_pyramid.residual_scale)
                * pyramid_residual
            )
        temporal = self.temporal_backbone(late_fusion)
        if self.evidence_preserving_block is not None:
            temporal = self.evidence_preserving_block(temporal)
        global_features = self._pool_features(temporal)
        return {
            "global": global_features,
            "temporal": temporal,
            "stage1_real": stage1[0],
            "stage1_imag": stage1[1],
            "stage2_real": stage2[0],
            "stage2_imag": stage2[1],
            "stage3_real": stage3[0],
            "stage3_imag": stage3[1],
            "stage4_real": stage4[0],
            "stage4_imag": stage4[1],
            "cross_stage_pyramid": pyramid_residual,
        }

    def forward_temporal_features(self, values):
        self._validate_input(values)
        _, stage2, stage3, stage4 = self.complex_encoder.forward_pyramid(
            values[:, 0:1, :], values[:, 1:2, :]
        )
        temporal = self.late_fusion(*stage4)
        if self.cross_stage_pyramid is not None:
            temporal = self.cross_stage_pyramid(
                temporal, stage2, stage3, stage4
            )
        temporal = self.temporal_backbone(temporal)
        if self.evidence_preserving_block is not None:
            temporal = self.evidence_preserving_block(temporal)
        return temporal

    def forward_feature_branches(self, values):
        """Return full F2 and phase-free Cartesian/magnitude pooled features.

        The second branch reuses every frozen module except the phase residual,
        making it suitable for a low-weight SawFM expert without introducing a
        second independently trained encoder.
        """

        self._validate_input(values)
        _, stage2, stage3, stage4 = self.complex_encoder.forward_pyramid(
            values[:, 0:1, :], values[:, 1:2, :]
        )
        real, imag = stage4
        cartesian, stable_magnitude, phase = self.late_fusion.forward_components(
            real, imag
        )
        full_fusion = self.late_fusion.combine_components(
            cartesian, stable_magnitude, phase
        )
        phase_free_fusion = self.late_fusion.combine_components(
            cartesian, stable_magnitude, None
        )
        if self.cross_stage_pyramid is not None:
            pyramid = self.cross_stage_pyramid.forward_pyramid(
                stage2, stage3, stage4
            )
            scale = self.cross_stage_pyramid.residual_bound * torch.tanh(
                self.cross_stage_pyramid.residual_scale
            )
            full_fusion = full_fusion + scale * pyramid
            phase_free_fusion = phase_free_fusion + scale * pyramid
        full_temporal = self.temporal_backbone(full_fusion)
        phase_free_temporal = self.temporal_backbone(phase_free_fusion)
        if self.evidence_preserving_block is not None:
            full_temporal = self.evidence_preserving_block(full_temporal)
            phase_free_temporal = self.evidence_preserving_block(
                phase_free_temporal
            )
        full_features = self._pool_features(full_temporal)
        phase_free_features = self._pool_features(phase_free_temporal)
        return full_features, phase_free_features

    def forward_features(self, values):
        temporal = self.forward_temporal_features(values)
        return self._pool_features(temporal)

    def forward_classifier(self, features):
        return self.classifier(features)

    def forward(self, values, return_features=False):
        features = self.forward_features(values)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits
