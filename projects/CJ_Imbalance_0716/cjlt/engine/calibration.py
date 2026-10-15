"""Temperature scaling restricted to the calibration partition."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    loss_before: float
    loss_after: float
    iterations: int
    source_role: str


def _require_calibration_role(source_role: str) -> None:
    if str(source_role) != "calibration":
        raise ValueError(
            "Temperature fitting requires source_role='calibration', got {!r}".format(
                source_role
            )
        )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    scale = float(temperature)
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("temperature must be finite and positive")
    return (values / scale).astype(np.float32, copy=False)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    source_role: str,
    max_iter: int = 50,
) -> TemperatureCalibration:
    _require_calibration_role(source_role)
    logit_values = np.asarray(logits, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.float32)
    if logit_values.shape != label_values.shape or logit_values.ndim != 2:
        raise ValueError(
            "calibration logits and labels must be matching two-dimensional arrays"
        )
    if len(logit_values) == 0:
        raise ValueError("calibration partition must not be empty")

    torch_logits = torch.from_numpy(logit_values)
    torch_labels = torch.from_numpy(label_values)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_before = float(criterion(torch_logits, torch_labels).item())
    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(torch.clamp(log_temperature, min=-5.0, max=5.0))
        loss = criterion(torch_logits / temperature, torch_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        log_temperature.clamp_(min=-5.0, max=5.0)
        temperature_value = float(torch.exp(log_temperature).item())
        loss_after = float(
            criterion(torch_logits / temperature_value, torch_labels).item()
        )
    if not np.isfinite(loss_after) or loss_after > loss_before:
        temperature_value = 1.0
        loss_after = loss_before
    return TemperatureCalibration(
        temperature=temperature_value,
        loss_before=loss_before,
        loss_after=loss_after,
        iterations=int(max_iter),
        source_role="calibration",
    )

