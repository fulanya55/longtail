"""Controlled I/Q interface models: complex convolution vs. grouped I/Q convolution."""

from __future__ import annotations

import importlib.util
import math
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


# Legacy backbone definitions live beside this project under LJQ_code.  Derive
# the location from this installed package so a relocated project never falls
# back to a historical drive letter or the caller's working directory.
LEGACY_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "pythonProject" / "新建文件夹"
LEGACY_MODEL_NAMES = (
    "legacy_mlcnn",
    "legacy_resnet34",
    "legacy_shufflenetv2",
    "legacy_mobilenetv2",
    "legacy_effnet_v1",
    "legacy_effnet_v2",
    "legacy_effnet_v3",
    "legacy_effnet_2d_iqm",
)
STANDARD_MODEL_NAMES = (
    "efficientnet_b0_1d",
    "efficientnet_b1_1d",
    "efficientnet_b1_moe3_1d",
    "efficientnet_b1_vft_residual_moe_1d",
    "convnext_tiny_1d",
)
MODEL_NAMES = ("complex1d", "iq_grouped") + STANDARD_MODEL_NAMES + LEGACY_MODEL_NAMES

DEFAULT_UNIFIED_HEAD_DIM = 256


@lru_cache(maxsize=None)
def _load_legacy_module(filename: str):
    """Load an unchanged legacy definition, resolving its local helper imports."""
    source = LEGACY_SOURCE_ROOT / filename
    if not source.is_file():
        raise FileNotFoundError(f"Legacy model source is missing: {source}")
    legacy_root = str(LEGACY_SOURCE_ROOT)
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    module_name = "cj0724_legacy_" + source.stem.lower()
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LegacyConcatIQAdapter(nn.Module):
    """Present `[I,Q]` as legacy `[B,1,I...Q]` without changing the old backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        return self.backbone(torch.cat([iq[:, 0:1], iq[:, 1:2]], dim=-1))


class LegacyEfficientNet2DIQAdapter(nn.Module):
    """Feed the old 2D EfficientNet with I, Q, and magnitude as its three channels."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        magnitude = torch.sqrt(iq[:, 0:1].square() + iq[:, 1:2].square() + 1.0e-8)
        # Legacy EffNet.py is a 3-channel 2D network. Height=1 treats the signal as a 1D image.
        image = torch.cat([iq[:, 0:1], iq[:, 1:2], magnitude], dim=1).unsqueeze(2)
        return self.backbone(image)


def _legacy_args(num_classes: int) -> SimpleNamespace:
    # The original `model.py` and `DBmodel.py` define data_point as total concatenated length.
    return SimpleNamespace(data_point=6000, num_classes=int(num_classes))


def _build_legacy_model(name: str, num_classes: int, dropout: float, attention: str) -> nn.Module:
    """Construct legacy backbones with their original internal structure and fresh heads."""
    args = _legacy_args(num_classes)
    if name == "legacy_mlcnn":
        return LegacyConcatIQAdapter(_load_legacy_module("model.py").cnnnet(args))
    legacy_1d = _load_legacy_module("DBmodel.py")
    if name == "legacy_resnet34":
        return LegacyConcatIQAdapter(legacy_1d.ResNet34_1D_Complex(args))
    if name == "legacy_shufflenetv2":
        return LegacyConcatIQAdapter(legacy_1d.ShuffleNetV2_1D_Complex(args, width_mult=1.0, dropout=dropout))
    if name == "legacy_mobilenetv2":
        return LegacyConcatIQAdapter(legacy_1d.MobileNetV2_1D_Complex(args, width_mult=1.0, dropout=dropout))
    if name == "legacy_effnet_v1":
        module = _load_legacy_module("EffNetV1.py")
        return LegacyConcatIQAdapter(module.EfficientNet(1.0, 1.0, 3000, num_classes, dropout, 0.2))
    if name == "legacy_effnet_v2":
        module = _load_legacy_module("EffNetV2.py")
        return LegacyConcatIQAdapter(module.EfficientNet(1.0, 1.0, 3000, num_classes, dropout, 0.2))
    if name == "legacy_effnet_v3":
        module = _load_legacy_module("EffNetV3.py")
        return LegacyConcatIQAdapter(
            module.EfficientNetV3(1.0, 1.0, 3000, num_classes, dropout, 0.2, attention=attention)
        )
    if name == "legacy_effnet_2d_iqm":
        module = _load_legacy_module("EffNet.py")
        return LegacyEfficientNet2DIQAdapter(module.EfficientNet(1.0, 1.0, num_classes, dropout, 0.2))
    raise ValueError(f"Unknown legacy model {name!r}")


class ComplexConv1d(nn.Module):
    """Complex 1D convolution using the exact real/imaginary product rule."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1, padding: int = 0):
        super().__init__()
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.real_bias = nn.Parameter(torch.zeros(out_channels))
        self.imag_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.real_conv(real) - self.imag_conv(imag)
        out_imag = self.real_conv(imag) + self.imag_conv(real)
        return (
            out_real + self.real_bias[None, :, None],
            out_imag + self.imag_bias[None, :, None],
        )


class ComplexBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True):
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.real_norm = nn.BatchNorm1d(out_channels)
        self.imag_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real, imag = self.conv(real, imag)
        return self.pool(F.gelu(self.real_norm(real))), self.pool(F.gelu(self.imag_norm(imag)))


class ComplexConvClassifier(nn.Module):
    """Three-block true-complex encoder with real, imaginary, and magnitude pooling."""

    def __init__(self, num_classes: int = 6, width: int = 24, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ComplexBlock1d(1, width, 7),
                ComplexBlock1d(width, width * 2, 5),
                ComplexBlock1d(width * 2, width * 4, 3),
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4 * 3, num_classes))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        real, imag = iq[:, 0:1], iq[:, 1:2]
        for block in self.blocks:
            real, imag = block(real, imag)
        magnitude = torch.sqrt(real.square() + imag.square() + 1.0e-8)
        features = torch.cat(
            [self.pool(real).squeeze(-1), self.pool(imag).squeeze(-1), self.pool(magnitude).squeeze(-1)], dim=1
        )
        return self.classifier(features)


class GroupedIQBlock1d(nn.Module):
    """Keep the I and Q routes separate with `groups=2` before cross-IQ fusion."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True):
        super().__init__()
        if in_channels % 2 or out_channels % 2:
            raise ValueError("Grouped I/Q blocks require even channel counts")
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(2) if pool else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IQGroupedConvClassifier(nn.Module):
    """Two I/Q grouped filter banks, then an explicit learnable 1x1 I/Q fusion."""

    def __init__(self, num_classes: int = 6, width: int = 24, dropout: float = 0.2):
        super().__init__()
        self.iq_paths = nn.Sequential(
            GroupedIQBlock1d(2, width * 2, 7),
            GroupedIQBlock1d(width * 2, width * 4, 5),
            GroupedIQBlock1d(width * 4, width * 8, 3),
        )
        self.fuse_iq = nn.Sequential(
            nn.Conv1d(width * 8, width * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, num_classes))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        features = self.fuse_iq(self.iq_paths(iq))
        return self.classifier(self.pool(features).squeeze(-1))


class WeakVisibilityResidualHead(nn.Module):
    """Bounded I/Q visibility correction on top of frozen classifier logits.

    The head consumes only the current I/Q view and the frozen logits available
    at deployment.  Equal-power reference views are therefore training-only
    supervision and cannot leak into inference.
    """

    def __init__(
        self,
        num_classes: int = 6,
        width: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        residual_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if residual_bound <= 0.0:
            raise ValueError("residual_bound must be positive")
        encoder = IQGroupedConvClassifier(num_classes=num_classes, width=width, dropout=0.0)
        encoder.classifier = nn.Identity()
        self.encoder = encoder
        feature_dim = int(width) * 4
        self.head = nn.Sequential(
            nn.Linear(feature_dim + int(num_classes), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.num_classes = int(num_classes)
        self.residual_bound = float(residual_bound)

    def forward_components(
        self, iq: torch.Tensor, frozen_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if frozen_logits.ndim != 2 or frozen_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected frozen logits [B, {self.num_classes}], got {tuple(frozen_logits.shape)}"
            )
        features = self.encoder(iq)
        raw_residual = self.head(torch.cat([features, frozen_logits.detach()], dim=1))
        residual = self.residual_bound * torch.tanh(raw_residual)
        return {
            "logits": frozen_logits.detach() + residual,
            "base_logits": frozen_logits.detach(),
            "residual": residual,
        }

    def forward(self, iq: torch.Tensor, frozen_logits: torch.Tensor) -> torch.Tensor:
        return self.forward_components(iq, frozen_logits)["logits"]


class SideRoutedPhysicsResidualHead(nn.Module):
    """Route fixed-weak and swept-weak corrections to physics-shaped experts."""

    def __init__(
        self,
        num_classes: int = 6,
        width: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        residual_bound: float = 2.5,
        maximum_period_lag: int = 768,
        maximum_delay_lag: int = 1200,
    ) -> None:
        super().__init__()
        from cj0724.physics_parameter_specialists import SequenceSummary

        if int(num_classes) != 6:
            raise ValueError("side-routed residual currently requires six classes")
        self.swept_sequence = SequenceSummary(3, width)
        self.swept_period = SequenceSummary(2, width)
        self.fixed_delay = SequenceSummary(3, width)
        base_dim = int(num_classes) + 5
        swept_dim = self.swept_sequence.feature_dim + self.swept_period.feature_dim
        fixed_dim = self.fixed_delay.feature_dim
        self.router = nn.Sequential(
            nn.Linear(base_dim, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.fixed_head = nn.Sequential(
            nn.Linear(fixed_dim + base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )
        self.swept_head = nn.Sequential(
            nn.Linear(swept_dim + base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.fixed_head[-1].weight)
        nn.init.zeros_(self.fixed_head[-1].bias)
        nn.init.zeros_(self.swept_head[-1].weight)
        nn.init.zeros_(self.swept_head[-1].bias)
        self.residual_bound = float(residual_bound)
        self.maximum_period_lag = int(maximum_period_lag)
        self.maximum_delay_lag = int(maximum_delay_lag)

    @staticmethod
    def _base_features(logits: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits.detach())
        fixed_mean = probability[:, :3].mean(1, keepdim=True)
        swept_mean = probability[:, 3:].mean(1, keepdim=True)
        fixed_max = probability[:, :3].amax(1, keepdim=True)
        swept_max = probability[:, 3:].amax(1, keepdim=True)
        contrast = fixed_mean - swept_mean
        return torch.cat(
            [logits.detach(), fixed_mean, swept_mean, fixed_max, swept_max, contrast], dim=1
        )

    @staticmethod
    def _phase_increment(iq: torch.Tensor) -> torch.Tensor:
        real, imag = iq[:, 0].float(), iq[:, 1].float()
        cross = real[:, :-1] * imag[:, 1:] - imag[:, :-1] * real[:, 1:]
        dot = real[:, :-1] * real[:, 1:] + imag[:, :-1] * imag[:, 1:]
        return torch.atan2(cross, dot) / torch.pi

    @staticmethod
    def _normalized_autocorrelation(sequence: torch.Tensor, maximum_lag: int) -> torch.Tensor:
        values = sequence.float() - sequence.float().mean(dim=-1, keepdim=True)
        length = int(values.shape[-1])
        fft_length = 1 << (2 * length - 1).bit_length()
        spectrum = torch.fft.rfft(values, n=fft_length, dim=-1)
        correlation = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_length, dim=-1)
        correlation = correlation[:, 1 : min(length, maximum_lag + 1)]
        return correlation / values.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _swept_features(self, iq: torch.Tensor) -> torch.Tensor:
        frequency = self._phase_increment(iq)
        smooth_short = F.avg_pool1d(frequency[:, None], 9, 1, 4).squeeze(1)
        smooth_long = F.avg_pool1d(frequency[:, None], 33, 1, 16).squeeze(1)
        sequence_feature = self.swept_sequence(
            torch.stack([frequency, smooth_short, smooth_long], dim=1)
        )
        correlation = self._normalized_autocorrelation(frequency, self.maximum_period_lag)
        correlation_smooth = F.avg_pool1d(correlation[:, None], 9, 1, 4).squeeze(1)
        period_feature = self.swept_period(
            torch.stack([correlation, correlation_smooth], dim=1)
        )
        return torch.cat([sequence_feature, period_feature], dim=1)

    def _fixed_features(self, iq: torch.Tensor) -> torch.Tensor:
        values = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        length = int(values.shape[-1])
        spectrum = torch.fft.fft(values, n=2 * length, dim=-1)
        correlation = torch.fft.ifft(spectrum * spectrum.conj(), dim=-1).abs()
        correlation = correlation[:, 1 : min(length, self.maximum_delay_lag + 1)]
        correlation = correlation / correlation.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        smooth_short = F.avg_pool1d(correlation[:, None], 7, 1, 3).squeeze(1)
        smooth_long = F.avg_pool1d(correlation[:, None], 31, 1, 15).squeeze(1)
        return self.fixed_delay(
            torch.stack([correlation, smooth_short, smooth_long], dim=1)
        )

    def forward_components(
        self, iq: torch.Tensor, frozen_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        base_feature = self._base_features(frozen_logits)
        route_logits = self.router(base_feature)
        route_probability = torch.softmax(route_logits, dim=1)
        fixed_raw = self.fixed_head(torch.cat([self._fixed_features(iq), base_feature], dim=1))
        swept_raw = self.swept_head(torch.cat([self._swept_features(iq), base_feature], dim=1))
        fixed_residual = self.residual_bound * torch.tanh(fixed_raw) * route_probability[:, 0:1]
        swept_residual = self.residual_bound * torch.tanh(swept_raw) * route_probability[:, 2:3]
        residual = torch.cat([fixed_residual, swept_residual], dim=1)
        return {
            "logits": frozen_logits.detach() + residual,
            "base_logits": frozen_logits.detach(),
            "residual": residual,
            "route_logits": route_logits,
            "route_probability": route_probability,
            "fixed_residual": fixed_residual,
            "swept_residual": swept_residual,
        }

    def forward(self, iq: torch.Tensor, frozen_logits: torch.Tensor) -> torch.Tensor:
        return self.forward_components(iq, frozen_logits)["logits"]


class DropPath(nn.Module):
    """Per-sample stochastic depth used by EfficientNet and ConvNeXt."""

    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.drop_probability
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(mask_shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep_probability


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class SqueezeExcitation1D(nn.Module):
    def __init__(self, input_channels: int, expanded_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeezed_channels = max(1, input_channels // int(squeeze_factor))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reduce = nn.Conv1d(expanded_channels, squeezed_channels, kernel_size=1)
        self.expand = nn.Conv1d(squeezed_channels, expanded_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.reduce(scale))
        scale = torch.sigmoid(self.expand(scale))
        return x * scale


class MBConv1D(nn.Module):
    """1D adaptation of the EfficientNet MBConv + SE block."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        drop_path: float,
    ):
        super().__init__()
        expanded_channels = input_channels * int(expand_ratio)
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv1d(input_channels, expanded_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(expanded_channels, eps=1.0e-3, momentum=0.01),
                    nn.SiLU(inplace=True),
                ]
            )
        else:
            expanded_channels = input_channels
        layers.extend(
            [
                nn.Conv1d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.BatchNorm1d(expanded_channels, eps=1.0e-3, momentum=0.01),
                nn.SiLU(inplace=True),
                SqueezeExcitation1D(input_channels, expanded_channels),
                nn.Conv1d(expanded_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(output_channels, eps=1.0e-3, momentum=0.01),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and input_channels == output_channels
        self.drop_path = DropPath(drop_path) if self.use_residual else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        return x + self.drop_path(output) if self.use_residual else output


class EfficientNet1D(nn.Module):
    """Standard EfficientNet B0/B1 stage schedule operating directly on I/Q channels."""

    _BASE_CONFIG = (
        # expand ratio, kernel, output channels, repeat count, first-block stride
        (1, 3, 16, 1, 1),
        (6, 3, 24, 2, 2),
        (6, 5, 40, 2, 2),
        (6, 3, 80, 3, 2),
        (6, 5, 112, 3, 1),
        (6, 5, 192, 4, 2),
        (6, 3, 320, 1, 1),
    )

    def __init__(
        self,
        num_classes: int = 6,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout: float = 0.2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        stem_channels = _make_divisible(32 * width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv1d(2, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_channels, eps=1.0e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        input_channels = stem_channels
        total_blocks = sum(int(math.ceil(repeats * depth_coefficient)) for _, _, _, repeats, _ in self._BASE_CONFIG)
        block_index = 0
        for expand_ratio, kernel_size, channels, repeats, stride in self._BASE_CONFIG:
            output_channels = _make_divisible(channels * width_coefficient)
            scaled_repeats = int(math.ceil(repeats * depth_coefficient))
            for repeat_index in range(scaled_repeats):
                current_stride = stride if repeat_index == 0 else 1
                stochastic_depth = drop_path_rate * block_index / max(1, total_blocks - 1)
                blocks.append(
                    MBConv1D(
                        input_channels,
                        output_channels,
                        expand_ratio,
                        kernel_size,
                        current_stride,
                        stochastic_depth,
                    )
                )
                input_channels = output_channels
                block_index += 1
        self.blocks = nn.Sequential(*blocks)
        top_channels = _make_divisible(1280 * width_coefficient)
        self.top = nn.Sequential(
            nn.Conv1d(input_channels, top_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(top_channels, eps=1.0e-3, momentum=0.01),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(top_channels, num_classes))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        features = self.top(self.blocks(self.stem(iq)))
        return self.classifier(self.pool(features).squeeze(-1))


class EfficientNetB1MultiExpert1D(nn.Module):
    """EfficientNet-B1 with a shared trunk and three independent high-level experts.

    Stages 1-6 are shared. Each expert owns stage 7, the 1280-channel top, and
    the same 256-dimensional classifier used by the single-backbone benchmark.
    The inference logit is the arithmetic mean of the expert logits.
    """

    def __init__(
        self,
        num_classes: int = 6,
        num_experts: int = 3,
        head_dim: int = DEFAULT_UNIFIED_HEAD_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(num_experts) < 2:
            raise ValueError("Multi-expert EfficientNet requires at least two experts")
        template = EfficientNet1D(
            num_classes=num_classes,
            width_coefficient=1.0,
            depth_coefficient=1.1,
            dropout=dropout,
        )
        scaled_repeats = [
            int(math.ceil(repeats * 1.1))
            for _, _, _, repeats, _ in EfficientNet1D._BASE_CONFIG
        ]
        shared_block_count = sum(scaled_repeats[:-1])
        blocks = list(template.blocks.children())
        self.stem = template.stem
        self.shared_blocks = nn.Sequential(*blocks[:shared_block_count])
        tail_template = nn.Sequential(*blocks[shared_block_count:])
        self.expert_tails = nn.ModuleList(
            [tail_template] + [deepcopy(tail_template) for _ in range(int(num_experts) - 1)]
        )
        self.expert_tops = nn.ModuleList(
            [template.top] + [deepcopy(template.top) for _ in range(int(num_experts) - 1)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1280, int(head_dim)),
                    nn.LayerNorm(int(head_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(head_dim), int(num_classes)),
                )
                for _ in range(int(num_experts))
            ]
        )
        self.num_experts = int(num_experts)

    def forward_experts(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        shared = self.shared_blocks(self.stem(iq))
        logits = []
        for tail, top, head in zip(self.expert_tails, self.expert_tops, self.expert_heads):
            features = self.pool(top(tail(shared))).squeeze(-1)
            logits.append(head(features))
        return torch.stack(logits, dim=1)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.forward_experts(iq).mean(dim=1)


class ChannelFirstLayerNorm1D(nn.Module):
    """LayerNorm over channels for `N,C,L` tensors, matching ConvNeXt downsampling."""

    def __init__(self, channels: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        return normalized * self.weight[None, :, None] + self.bias[None, :, None]


class ConvNeXtBlock1D(nn.Module):
    """1D ConvNeXt block: depthwise 7-tap conv, channel MLP, layer scale, residual."""

    def __init__(self, channels: int, drop_path: float = 0.0, layer_scale_init: float = 1.0e-6):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).transpose(1, 2)
        x = self.project(self.activation(self.expand(self.norm(x))))
        x = (x * self.gamma).transpose(1, 2)
        return residual + self.drop_path(x)


class ConvNeXtTiny1D(nn.Module):
    """ConvNeXt-Tiny dimensions/depths adapted from images to direct I/Q sequences."""

    def __init__(self, num_classes: int = 6, drop_path_rate: float = 0.1):
        super().__init__()
        dimensions = (96, 192, 384, 768)
        depths = (3, 3, 9, 3)
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(2, dimensions[0], kernel_size=4, stride=4),
                    ChannelFirstLayerNorm1D(dimensions[0]),
                )
            ]
        )
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    ChannelFirstLayerNorm1D(dimensions[index]),
                    nn.Conv1d(dimensions[index], dimensions[index + 1], kernel_size=2, stride=2),
                )
            )
        rates = torch.linspace(0, float(drop_path_rate), sum(depths)).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for channels, depth in zip(dimensions, depths):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock1D(channels, rates[cursor + i]) for i in range(depth)]))
            cursor += depth
        self.norm = nn.LayerNorm(dimensions[-1], eps=1.0e-6)
        self.classifier = nn.Linear(dimensions[-1], num_classes)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L] I/Q input, got {tuple(iq.shape)}")
        x = iq
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        features = self.norm(x.mean(dim=-1))
        return self.classifier(features)


class EfficientNetB1VFTResidualMoE1D(nn.Module):
    """B1 general classifier with bounded weak-positive/hard-negative VFT experts."""

    def __init__(
        self,
        num_classes: int = 6,
        head_dim: int = DEFAULT_UNIFIED_HEAD_DIM,
        dropout: float = 0.2,
        target_index: int = 4,
        residual_bound: float = 0.5,
    ) -> None:
        super().__init__()
        if int(num_classes) <= int(target_index):
            raise ValueError("target_index is outside the class dimension")
        backbone = EfficientNet1D(
            num_classes=num_classes,
            width_coefficient=1.0,
            depth_coefficient=1.1,
            dropout=dropout,
        )
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.feature_adapter = nn.Sequential(
            nn.Linear(1280, int(head_dim)),
            nn.LayerNorm(int(head_dim)),
            nn.GELU(),
        )
        self.general_classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_dim), int(num_classes)),
        )
        specialist_dim = max(32, int(head_dim) // 4)
        self.weak_positive_head = nn.Sequential(
            nn.Linear(int(head_dim), specialist_dim),
            nn.GELU(),
            nn.Linear(specialist_dim, 1),
        )
        self.hard_negative_head = nn.Sequential(
            nn.Linear(int(head_dim), specialist_dim),
            nn.GELU(),
            nn.Linear(specialist_dim, 1),
        )
        nn.init.zeros_(self.weak_positive_head[-1].weight)
        nn.init.zeros_(self.weak_positive_head[-1].bias)
        nn.init.zeros_(self.hard_negative_head[-1].weight)
        nn.init.zeros_(self.hard_negative_head[-1].bias)
        self.target_index = int(target_index)
        self.residual_bound = float(residual_bound)

    def forward_components(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_adapter(self.backbone(iq))
        general_logits = self.general_classifier(features)
        weak_positive_score = self.weak_positive_head(features).squeeze(-1)
        hard_negative_score = self.hard_negative_head(features).squeeze(-1)
        target_residual = self.residual_bound * (
            torch.tanh(weak_positive_score) - torch.tanh(hard_negative_score)
        )
        target_logit = general_logits[:, self.target_index] + target_residual
        fused_logits = torch.cat(
            [
                general_logits[:, : self.target_index],
                target_logit[:, None],
                general_logits[:, self.target_index + 1 :],
            ],
            dim=1,
        )
        return {
            "logits": fused_logits,
            "general_logits": general_logits,
            "weak_positive_score": weak_positive_score,
            "hard_negative_score": hard_negative_score,
            "target_residual": target_residual,
        }

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.forward_components(iq)["logits"]


def build_model(
    name: str,
    num_classes: int = 6,
    width: int = 24,
    dropout: float = 0.2,
    head_dim: int = DEFAULT_UNIFIED_HEAD_DIM,
    legacy_attention: str = "se",
) -> nn.Module:
    canonical = str(name).lower()
    if canonical == "complex1d":
        return ComplexConvClassifier(num_classes=num_classes, width=width, dropout=dropout)
    if canonical == "iq_grouped":
        return IQGroupedConvClassifier(num_classes=num_classes, width=width, dropout=dropout)
    if canonical == "efficientnet_b0_1d":
        return EfficientNet1D(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout=0.2)
    if canonical == "efficientnet_b1_1d":
        # B1's original image-resolution change has no 1D counterpart; retain 3000 samples and apply its depth scale.
        return EfficientNet1D(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.1, dropout=0.2)
    if canonical == "efficientnet_b1_moe3_1d":
        return EfficientNetB1MultiExpert1D(
            num_classes=num_classes,
            num_experts=3,
            head_dim=head_dim,
            dropout=dropout,
        )
    if canonical == "efficientnet_b1_vft_residual_moe_1d":
        return EfficientNetB1VFTResidualMoE1D(
            num_classes=num_classes,
            head_dim=head_dim,
            dropout=dropout,
        )
    if canonical == "convnext_tiny_1d":
        return ConvNeXtTiny1D(num_classes=num_classes)
    if canonical in LEGACY_MODEL_NAMES:
        return _build_legacy_model(canonical, num_classes, dropout, legacy_attention)
    raise ValueError("Unknown model {!r}; choose one of {}".format(name, ", ".join(MODEL_NAMES)))


class UnifiedBackboneClassifier(nn.Module):
    """Standardize every backbone to a common embedding and identical classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 6,
        head_dim: int = DEFAULT_UNIFIED_HEAD_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.raw_feature_dim = int(feature_dim)
        self.head_dim = int(head_dim)
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.raw_feature_dim, self.head_dim),
            nn.LayerNorm(self.head_dim),
            nn.GELU(),
        )
        # This module is exactly identical for every backbone in the benchmark.
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.head_dim, int(num_classes)),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        features = self.backbone(iq)
        if features.ndim != 2 or features.shape[1] != self.raw_feature_dim:
            raise RuntimeError(
                f"Backbone returned {tuple(features.shape)}; expected [B, {self.raw_feature_dim}]"
            )
        return self.classifier(self.feature_adapter(features))


def _strip_native_or_legacy_classifier(
    model: nn.Module,
    name: str,
    width: int,
    common_dropout: float,
) -> tuple[nn.Module, int]:
    """Remove model-specific heads and return the raw feature dimension."""
    if name == "complex1d":
        model.classifier = nn.Identity()
        return model, int(width) * 4 * 3
    if name == "iq_grouped":
        model.classifier = nn.Identity()
        return model, int(width) * 4
    if name in ("efficientnet_b0_1d", "efficientnet_b1_1d"):
        model.classifier = nn.Identity()
        return model, 1280
    if name == "convnext_tiny_1d":
        model.classifier = nn.Identity()
        return model, 768
    if name == "legacy_mlcnn":
        model.backbone.fc2 = nn.Identity()
        model.backbone.drop.p = float(common_dropout)
        return model, 512
    if name == "legacy_resnet34":
        model.backbone.classifier = nn.Identity()
        model.backbone.dropout.p = float(common_dropout)
        return model, 1024
    if name == "legacy_shufflenetv2":
        model.backbone.classifier = nn.Identity()
        model.backbone.dropout.p = float(common_dropout)
        return model, 2048
    if name == "legacy_mobilenetv2":
        model.backbone.classifier = nn.Identity()
        model.backbone.dropout.p = float(common_dropout)
        return model, 2560
    if name == "legacy_effnet_v1":
        model.backbone.classifier = nn.Identity()
        return model, 10240
    if name == "legacy_effnet_v2":
        model.backbone.classifier = nn.Identity()
        return model, 20480
    if name == "legacy_effnet_v3":
        model.backbone.classifier = nn.Identity()
        return model, 10240
    if name == "legacy_effnet_2d_iqm":
        model.backbone.classifier = nn.Identity()
        return model, 1280
    raise ValueError(f"No feature adapter is registered for {name!r}")


def build_benchmark_model(
    name: str,
    num_classes: int = 6,
    width: int = 24,
    dropout: float = 0.2,
    head_dim: int = DEFAULT_UNIFIED_HEAD_DIM,
    legacy_attention: str = "se",
) -> nn.Module:
    """Build a backbone with its native head removed and the common benchmark head attached."""
    canonical = str(name).lower()
    if canonical == "efficientnet_b1_moe3_1d":
        return build_model(
            canonical,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            head_dim=head_dim,
            legacy_attention=legacy_attention,
        )
    if canonical == "efficientnet_b1_vft_residual_moe_1d":
        return build_model(
            canonical,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            head_dim=head_dim,
            legacy_attention=legacy_attention,
        )
    model = build_model(
        canonical,
        num_classes=num_classes,
        width=width,
        dropout=dropout,
        head_dim=head_dim,
        legacy_attention=legacy_attention,
    )
    backbone, feature_dim = _strip_native_or_legacy_classifier(
        model,
        canonical,
        width,
        dropout,
    )
    return UnifiedBackboneClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        head_dim=head_dim,
        dropout=dropout,
    )
