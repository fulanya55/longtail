"""Oracle-positive parameter specialists for identifiability experiments."""

import torch
import torch.nn.functional as F
from torch import nn

from cjlt.models.backbone_adapter import LightweightComplexIQClassifier


class OrdinalCountHead(nn.Module):
    """Five ordered thresholds for integer counts 3 through 8."""

    def __init__(self, feature_dim):
        super().__init__()
        self.score = nn.Linear(int(feature_dim), 1)
        self.threshold_start = nn.Parameter(torch.tensor(-1.0))
        self.threshold_gaps = nn.Parameter(torch.zeros(4))

    def thresholds(self):
        gaps = F.softplus(self.threshold_gaps) + 1.0e-3
        return torch.cat(
            [self.threshold_start[None], self.threshold_start[None] + gaps.cumsum(0)]
        )

    def forward(self, features):
        score = self.score(features)
        logits = score - self.thresholds()[None, :]
        expected_count = 3.0 + torch.sigmoid(logits).sum(dim=-1)
        return logits, expected_count


def ordinal_count_loss(logits, raw_count):
    target_class = (raw_count.round().long() - 3).clamp(0, 5)
    levels = torch.arange(5, device=logits.device)[None, :]
    cumulative = (target_class[:, None] > levels).float()
    return F.binary_cross_entropy_with_logits(logits.float(), cumulative)


class RawIQEncoder(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.backbone = LightweightComplexIQClassifier(
            num_classes=1, width=int(width), dropout=0.0
        )
        self.feature_dim = int(self.backbone.feature_dim)

    def forward(self, signal):
        return self.backbone.forward_features(signal)


class MFTAutocorrelationEncoder(nn.Module):
    """Expose delayed-copy peaks through linear autocorrelation and top-k lags."""

    def __init__(self, width=16, maximum_lag=1500, peak_count=16):
        super().__init__()
        base = int(width)
        self.maximum_lag = int(maximum_lag)
        self.peak_count = int(peak_count)
        self.convolution = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base * 2),
            nn.GELU(),
            nn.Conv1d(
                base * 2, base * 4, kernel_size=5, stride=2, padding=2, bias=False
            ),
            nn.BatchNorm1d(base * 4),
            nn.GELU(),
        )
        self.feature_dim = base * 8 + self.peak_count * 2

    def _autocorrelation(self, signal):
        iq = signal.float()
        complex_signal = torch.complex(iq[:, 0], iq[:, 1])
        length = int(complex_signal.shape[-1])
        spectrum = torch.fft.fft(complex_signal, n=2 * length, dim=-1)
        correlation = torch.fft.ifft(spectrum * spectrum.conj(), dim=-1).abs()
        correlation = correlation[:, 1 : min(self.maximum_lag + 1, length)]
        scale = correlation.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return correlation / scale

    def forward(self, signal):
        correlation = self._autocorrelation(signal)
        temporal = self.convolution(correlation[:, None, :])
        average = F.adaptive_avg_pool1d(temporal, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(temporal, 1).squeeze(-1)
        smoothed = F.avg_pool1d(
            correlation[:, None, :], kernel_size=7, stride=1, padding=3
        ).squeeze(1)
        count = min(self.peak_count, int(smoothed.shape[-1]))
        peak_values, peak_indices = torch.topk(smoothed, k=count, dim=-1)
        peak_indices = peak_indices.float() / float(max(smoothed.shape[-1] - 1, 1))
        if count < self.peak_count:
            padding = self.peak_count - count
            peak_values = F.pad(peak_values, (0, padding))
            peak_indices = F.pad(peak_indices, (0, padding))
        peak_indices, order = peak_indices.sort(dim=-1)
        peak_values = peak_values.gather(1, order)
        return torch.cat([average, maximum, peak_values, peak_indices], dim=1)


class MFTOrdinalSpecialist(nn.Module):
    target_name = "MFT"

    def __init__(self, encoder="raw", width=16):
        super().__init__()
        normalized = str(encoder).lower()
        if normalized == "raw":
            self.encoder = RawIQEncoder(width=width)
        elif normalized == "autocorrelation":
            self.encoder = MFTAutocorrelationEncoder(width=width)
        else:
            raise ValueError("Unknown MFT encoder: {!r}".format(encoder))
        self.ordinal_head = OrdinalCountHead(self.encoder.feature_dim)

    def forward(self, signal):
        logits, prediction = self.ordinal_head(self.encoder(signal))
        return {"ordinal_logits": logits, "prediction": prediction}


class ScalarRegressionHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class VFTSTFTEncoder(nn.Module):
    """Preserve the time axis and summarize both the spectrogram and its ridge."""

    def __init__(self, width=8, n_fft=128, hop_length=32):
        super().__init__()
        base = int(width)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.spectrogram = nn.Sequential(
            nn.Conv2d(1, base, 5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.GELU(),
        )
        self.ridge = nn.Sequential(
            nn.Conv1d(3, base * 2, 5, padding=2, bias=False),
            nn.BatchNorm1d(base * 2),
            nn.GELU(),
            nn.Conv1d(base * 2, base * 4, 5, padding=2, bias=False),
            nn.BatchNorm1d(base * 4),
            nn.GELU(),
        )
        self.feature_dim = base * 16

    def forward(self, signal):
        iq = signal.float()
        values = torch.complex(iq[:, 0], iq[:, 1])
        spectrum = torch.stft(
            values,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(values.device),
            center=True,
            return_complex=True,
            onesided=False,
        )
        power = torch.fft.fftshift(spectrum.abs().square(), dim=1)
        distribution = power / power.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        log_power = torch.log1p(distribution * float(self.n_fft))
        image = self.spectrogram(log_power[:, None])
        image_average = F.adaptive_avg_pool2d(image, 1).flatten(1)
        image_maximum = F.adaptive_max_pool2d(image, 1).flatten(1)
        coordinate = torch.linspace(
            -1.0, 1.0, self.n_fft, device=power.device, dtype=power.dtype
        )
        centroid = (distribution * coordinate[None, :, None]).sum(dim=1)
        spread = (
            distribution
            * (coordinate[None, :, None] - centroid[:, None, :]).square()
        ).sum(dim=1).sqrt()
        peak = coordinate[distribution.argmax(dim=1)]
        ridge = self.ridge(torch.stack([centroid, spread, peak], dim=1))
        ridge_average = F.adaptive_avg_pool1d(ridge, 1).squeeze(-1)
        ridge_maximum = F.adaptive_max_pool1d(ridge, 1).squeeze(-1)
        return torch.cat(
            [image_average, image_maximum, ridge_average, ridge_maximum], dim=1
        )


class VFTInstantaneousFrequencyEncoder(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        base = int(width)
        self.temporal = nn.Sequential(
            nn.Conv1d(3, base, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base * 2),
            nn.GELU(),
            nn.Conv1d(base * 2, base * 4, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(base * 4),
            nn.GELU(),
        )
        self.feature_dim = base * 8

    def forward(self, signal):
        iq = signal.float()
        values = torch.complex(iq[:, 0], iq[:, 1])
        phase_increment = torch.angle(values[:, 1:] * values[:, :-1].conj()) / torch.pi
        magnitude = values.abs()
        amplitude = 0.5 * (magnitude[:, 1:] + magnitude[:, :-1])
        amplitude_change = magnitude[:, 1:] - magnitude[:, :-1]
        features = torch.stack([phase_increment, amplitude, amplitude_change], dim=1)
        temporal = self.temporal(features)
        return torch.cat(
            [
                F.adaptive_avg_pool1d(temporal, 1).squeeze(-1),
                F.adaptive_max_pool1d(temporal, 1).squeeze(-1),
            ],
            dim=1,
        )


class VFTDopplerSpecialist(nn.Module):
    target_name = "VFT"

    def __init__(self, encoder="raw", width=16):
        super().__init__()
        normalized = str(encoder).lower()
        if normalized == "raw":
            self.encoder = RawIQEncoder(width=width)
        elif normalized == "stft":
            self.encoder = VFTSTFTEncoder(width=max(int(width) // 2, 4))
        elif normalized in ("instantaneous_frequency", "ifreq"):
            self.encoder = VFTInstantaneousFrequencyEncoder(width=width)
        else:
            raise ValueError("Unknown VFT encoder: {!r}".format(encoder))
        self.regression_head = ScalarRegressionHead(self.encoder.feature_dim)

    def forward(self, signal):
        return {"prediction": self.regression_head(self.encoder(signal))}


def build_oracle_specialist(variant, width=16):
    variants = {
        "MFT_RAW_ORD": (MFTOrdinalSpecialist, "raw"),
        "MFT_ACF_ORD": (MFTOrdinalSpecialist, "autocorrelation"),
        "VFT_RAW": (VFTDopplerSpecialist, "raw"),
        "VFT_STFT": (VFTDopplerSpecialist, "stft"),
        "VFT_IFREQ": (VFTDopplerSpecialist, "instantaneous_frequency"),
    }
    try:
        model_class, encoder = variants[str(variant)]
    except KeyError:
        raise ValueError("Unknown oracle specialist variant: {!r}".format(variant))
    return model_class(encoder=encoder, width=int(width))
