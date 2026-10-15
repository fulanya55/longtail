"""Auditable experiment snapshots, event streams, and failure receipts."""

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys

import torch

from cjlt.engine.run_manager import configuration_sha256


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Unsupported record value: {!r}".format(type(value)))


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )


class ExperimentRecord:
    def __init__(self, output_dir, scientific_config):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_sha256 = configuration_sha256(scientific_config)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.write_environment()
        write_json(
            self.output_dir / "scientific_config.json",
            {
                "config_sha256": self.config_sha256,
                "started_at_utc": self.started_at,
                "scientific_config": scientific_config,
            },
        )
        self.event("run_started", {"config_sha256": self.config_sha256})

    def write_environment(self):
        cuda_name = None
        if torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(0)
        write_json(
            self.output_dir / "environment.json",
            {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": torch.version.cuda,
                "cuda_device": cuda_name,
            },
        )

    def event(self, name, payload=None):
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": str(name),
            "payload": payload or {},
        }
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )

    def complete(self, receipt):
        payload = dict(receipt)
        payload.update(
            {
                "status": "completed",
                "config_sha256": self.config_sha256,
                "started_at_utc": self.started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(self.output_dir / "run_receipt.json", payload)
        self.event("run_completed", payload)
        return payload

    def fail(self, error, context=None):
        payload = {
            "status": "failed",
            "config_sha256": self.config_sha256,
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {},
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(self.output_dir / "failure.json", payload)
        write_json(self.output_dir / "run_receipt.json", payload)
        self.event("run_failed", payload)
        return payload
