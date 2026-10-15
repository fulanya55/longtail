import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cj0724.audit_p5r_relocation_safe_data import historical_candidates, validate_historical_baseline


def test_p5r_baseline_is_explicit_hash_pinned_and_p5q_runner_accepts_protocol(tmp_path):
    project = Path(__file__).resolve().parents[1]
    sample = tmp_path / "baseline.mat"
    sample.write_bytes(b"p5r-baseline")
    digest = hashlib.sha256(sample.read_bytes()).hexdigest()
    receipt = validate_historical_baseline(sample, digest)
    assert Path(receipt["path"]) == sample
    with pytest.raises(RuntimeError):
        validate_historical_baseline(sample, "0" * 64)
    candidates = historical_candidates(project, sample)
    assert sample.resolve() in candidates
    assert not any(str(path).startswith("C:\\LJQ_code") for path in candidates)

    protocol_path = project / "configs/p5r_explicit_baseline_audit_preregister_20260805.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project)
    command = [sys.executable, "-m", "cj0724.p5q_relocation_safe_checkpoint_execution", "--preregister", str(protocol_path), "--p5e-freeze", str(project / protocol["base_candidate"]["freeze_path"]), "--validate-protocol"]
    completed = subprocess.run(command, cwd=unrelated, env=environment, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["protocol_valid"] is True
