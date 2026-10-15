"""Hash-safe run recovery and error isolation helpers."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


class RunConfigurationConflict(RuntimeError):
    pass


@dataclass
class RunDecision:
    action: str
    receipt: object = None


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Unsupported configuration value: {!r}".format(type(value)))


def configuration_sha256(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_run_arguments(arguments):
    """Remove orchestration-only values from a scientific config hash."""

    excluded = {"resume", "dry_run", "output_dir"}
    return {key: value for key, value in dict(arguments).items() if key not in excluded}


def decide_run(output_dir, config_hash, resume=False):
    directory = Path(output_dir)
    receipt_path = directory / "run_receipt.json"
    if not receipt_path.is_file():
        if directory.is_dir() and any(directory.iterdir()):
            raise RunConfigurationConflict(
                "Existing nonempty run directory has no receipt: {}".format(directory)
            )
        return RunDecision("run")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    existing_hash = receipt.get("config_sha256")
    if existing_hash != str(config_hash):
        raise RunConfigurationConflict(
            "Run directory config hash mismatch: {} != {}".format(
                existing_hash, config_hash
            )
        )
    if receipt.get("status") == "completed" and bool(resume):
        return RunDecision("skip", receipt=receipt)
    raise RunConfigurationConflict(
        "Run directory already exists; pass --resume only for a matching completed run"
    )


def append_error(path, job, error):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "job": job,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
