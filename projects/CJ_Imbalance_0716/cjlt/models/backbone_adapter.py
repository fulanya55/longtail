"""Feature-compatible I/Q classifier derived from the validated CCEL interface."""

import torch
from torch import nn
from torch.nn import functional as F

from cjlt.models.complex_layers import ComplexConv1d, complex_magnitude
from cjlt.models.phase_multiscale import (
    DelayedFusionMultiScaleIQClassifier,
    PhaseAwareMultiScaleIQClassifier,
)


class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values):
        return self.block(values)


class LightweightComplexIQClassifier(nn.Module):
    """Complex stem plus shared real temporal backbone and original logits head."""

    def __init__(self, num_classes=6, width=16, dropout=0.2):
        super().__init__()
        self.input_mode = "complex_1d_true"
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.feature_dim = self.width * 4
        self.complex_stem = ComplexConv1d(
            1, self.width, kernel_size=7, padding=3, bias=False
        )
        self.complex_real_bn = nn.BatchNorm1d(self.width)
        self.complex_imag_bn = nn.BatchNorm1d(self.width)
        self.temporal_backbone = nn.Sequential(
            nn.MaxPool1d(2),
            ConvBnRelu1d(
                self.width, self.width * 2, kernel_size=5, padding=2
            ),
            nn.MaxPool1d(2),
            ConvBnRelu1d(
                self.width * 2, self.width * 4, kernel_size=3, padding=1
            ),
            nn.MaxPool1d(2),
            ConvBnRelu1d(
                self.width * 4, self.width * 4, kernel_size=3, padding=1
            ),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, self.num_classes),
        )

    def _complex_stem_features(self, values):
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(
                "complex_1d_true expects [batch, 2, length], got {}".format(
                    tuple(values.shape)
                )
            )
        real = values[:, 0:1, :]
        imag = values[:, 1:2, :]
        real, imag = self.complex_stem(real, imag)
        real = F.relu(self.complex_real_bn(real), inplace=True)
        imag = F.relu(self.complex_imag_bn(imag), inplace=True)
        return complex_magnitude(real, imag)

    def forward_temporal_features(self, values):
        return self.temporal_backbone(self._complex_stem_features(values))

    def forward_features(self, values):
        temporal = self.forward_temporal_features(values)
        return self.global_pool(temporal).squeeze(-1)

    def forward_classifier(self, features):
        return self.classifier(features)

    def forward(self, values, return_features=False):
        features = self.forward_features(values)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits


def build_backbone(
    model_name,
    num_classes,
    width=16,
    dropout=0.2,
):
    normalized = str(model_name).strip().lower()
    aliases = {
        "complex_1d_true": "complex_1d_true",
        "lightweight_complex": "complex_1d_true",
        "tiny_complex": "complex_1d_true",
        "phase_multiscale": "phase_multiscale",
        "phase_aware_multiscale": "phase_multiscale",
        "pam_iq": "phase_multiscale",
        "pam_iq_no_phase": "phase_multiscale_no_phase",
        "pam_iq_no_attention": "phase_multiscale_no_attention",
        "pam_iq_avg_pool": "phase_multiscale_avg_pool",
        "pam_iq_dual_pool": "phase_multiscale_dual_pool",
        "pam_iq_dual_query": "phase_multiscale_dual_query",
        "pam_iq_positional_query": "phase_multiscale_positional_query",
        "dfms_iq": "dfms_iq",
        "delayed_fusion_multiscale": "dfms_iq",
        "dfms_iq_no_phase": "dfms_iq_no_phase",
        "dfms_iq_anti_alias": "dfms_iq_anti_alias",
        "dfms_iq_m1": "dfms_iq_anti_alias",
        "dfms_iq_csfp": "dfms_iq_csfp",
        "dfms_iq_m2": "dfms_iq_csfp",
        "dfms_iq_epb": "dfms_iq_epb",
        "dfms_iq_m3": "dfms_iq_epb",
        "dfms_iq_local_topk8": "dfms_iq_local_topk8",
        "dfms_iq_local_topk16": "dfms_iq_local_topk16",
        "dfms_iq_m4_k8": "dfms_iq_local_topk8",
        "dfms_iq_m4_k16": "dfms_iq_local_topk16",
    }
    if normalized not in aliases:
        raise ValueError("Unknown backbone: {!r}".format(model_name))
    if aliases[normalized].startswith("dfms_iq"):
        variant = aliases[normalized]
        return DelayedFusionMultiScaleIQClassifier(
            num_classes=int(num_classes),
            width=int(width),
            dropout=float(dropout),
            include_phase=variant != "dfms_iq_no_phase",
            phase_lags=(1, 4, 16),
            anti_aliased_downsampling=variant == "dfms_iq_anti_alias",
            cross_stage_pyramid=variant == "dfms_iq_csfp",
            evidence_preserving_block=variant == "dfms_iq_epb",
            local_statistics_topk=(
                8
                if variant == "dfms_iq_local_topk8"
                else 16
                if variant == "dfms_iq_local_topk16"
                else None
            ),
        )
    if aliases[normalized].startswith("phase_multiscale"):
        variant = aliases[normalized]
        return PhaseAwareMultiScaleIQClassifier(
            num_classes=int(num_classes),
            width=int(width),
            dropout=float(dropout),
            include_phase_delta=variant != "phase_multiscale_no_phase",
            use_sparse_attention=variant != "phase_multiscale_no_attention",
            attentive_statistics_pooling=variant != "phase_multiscale_avg_pool",
            pooling_mode=(
                "dual_pool"
                if variant == "phase_multiscale_dual_pool"
                else "label_query"
                if variant == "phase_multiscale_dual_query"
                else "positional_label_query"
                if variant == "phase_multiscale_positional_query"
                else "single"
            ),
            residual_attention=variant in (
                "phase_multiscale_dual_pool",
                "phase_multiscale_dual_query",
                "phase_multiscale_positional_query",
            ),
            query_dim=32,
            attention_floor=0.2,
        )
    return LightweightComplexIQClassifier(
        num_classes=int(num_classes),
        width=int(width),
        dropout=float(dropout),
    )


def count_trainable_parameters(model):
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
