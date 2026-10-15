"""Training and metrics for oracle-positive single-parameter specialists."""

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from cjlt.models.oracle_parameter_specialists import ordinal_count_loss


@dataclass
class ScalarTargetNormalizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values):
        target = np.asarray(values, dtype=np.float64)
        return cls(float(target.mean()), float(max(target.std(), 1.0e-6)))

    def transform(self, values):
        return (values - float(self.mean)) / float(self.scale)

    def inverse_transform(self, values):
        return values * float(self.scale) + float(self.mean)


def _batch_target(batch, parameter_column, device):
    values = batch["parameter_values"][:, int(parameter_column)].to(device)
    mask = batch["parameter_mask"][:, int(parameter_column)].to(device).bool()
    if not bool(mask.all()):
        raise ValueError("oracle-positive loader contains an invalid parameter target")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("oracle-positive parameter targets must be finite")
    return values.float()


def specialist_predictions(
    model,
    loader,
    device,
    parameter_column,
    target_kind,
    normalizer=None,
    use_amp=False,
):
    model.eval()
    predictions = []
    targets = []
    sample_ids = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device, non_blocking=True)
            target = _batch_target(batch, parameter_column, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(signal)
            prediction = output["prediction"].float()
            if str(target_kind) == "regression":
                if normalizer is None:
                    raise ValueError("regression prediction requires a normalizer")
                prediction = normalizer.inverse_transform(prediction)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            sample_ids.append(
                torch.as_tensor(batch["sample_index"], dtype=torch.int64).numpy()
            )
    if not predictions:
        raise ValueError("specialist prediction loader must not be empty")
    return {
        "predictions": np.concatenate(predictions).astype(np.float32),
        "targets": np.concatenate(targets).astype(np.float32),
        "sample_ids": np.concatenate(sample_ids).astype(np.int64),
    }


def compute_specialist_metrics(predictions, targets, training_median, target_kind):
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    error = np.abs(prediction - target)
    constant_error = np.abs(float(training_median) - target)
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if len(target) > 2 and prediction.std() > 0.0 and target.std() > 0.0
        else 0.0
    )
    result = {
        "count": int(len(target)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson_correlation": correlation,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "training_median": float(training_median),
        "constant_median_mae": float(constant_error.mean()),
        "relative_mae_reduction_vs_constant": float(
            1.0 - error.mean() / max(constant_error.mean(), 1.0e-12)
        ),
    }
    if str(target_kind) == "ordinal":
        rounded = np.clip(np.rint(prediction), 3, 8)
        result.update(
            {
                "rounded_mae": float(np.abs(rounded - target).mean()),
                "rounded_accuracy": float((rounded == target).mean()),
            }
        )
    return result


def train_oracle_specialist(
    model,
    train_loader,
    validation_loader,
    device,
    parameter_column,
    target_kind,
    normalizer,
    training_median,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    checkpoint_path=None,
    use_amp=False,
    gradient_clip=5.0,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_epoch = 0
    best_mae = math.inf
    bad_epochs = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            signal = batch["signal"].to(device, non_blocking=True)
            raw_target = _batch_target(batch, parameter_column, device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(signal)
                if str(target_kind) == "ordinal":
                    loss = ordinal_count_loss(output["ordinal_logits"], raw_target)
                elif str(target_kind) == "regression":
                    normalized_target = normalizer.transform(raw_target)
                    loss = F.smooth_l1_loss(
                        output["prediction"].float(), normalized_target.float()
                    )
                else:
                    raise ValueError("Unknown target kind: {!r}".format(target_kind))
            scaler.scale(loss).backward()
            if float(gradient_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(signal)
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            sample_count += batch_size

        validation = specialist_predictions(
            model,
            validation_loader,
            device,
            parameter_column,
            target_kind,
            normalizer=normalizer,
            use_amp=use_amp,
        )
        metrics = compute_specialist_metrics(
            validation["predictions"],
            validation["targets"],
            training_median,
            target_kind,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": loss_sum / max(sample_count, 1),
                "validation_mae": metrics["mae"],
                "validation_correlation": metrics["pearson_correlation"],
                "validation_reduction_vs_constant": metrics[
                    "relative_mae_reduction_vs_constant"
                ],
                "epoch_seconds": float(time.perf_counter() - epoch_started),
            }
        )
        if float(metrics["mae"]) < best_mae:
            best_mae = float(metrics["mae"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(patience) > 0 and bad_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("specialist training produced no checkpoint")
    model.load_state_dict(best_state)
    result = {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_mae": best_mae,
        "train_seconds": float(time.perf_counter() - started),
        "history": history,
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": best_state, **result},
            str(path),
        )
    return result
