import json
import os
import subprocess
import sys
from pathlib import Path


def test_p5t_legacy_source_and_five_frozen_models_are_portable_without_data_or_cuda(tmp_path):
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/p5t_legacy_portable_preregister_20260806.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(project)
    unrelated = tmp_path / "unrelated"; unrelated.mkdir()
    common = [sys.executable, "-m", "cj0724.p5t_legacy_portable_execution", "--preregister", str(protocol_path), "--p5e-freeze", str(project / protocol["base_candidate"]["freeze_path"])]
    validation = json.loads(subprocess.run([*common, "--validate-protocol"], cwd=unrelated, env=environment, capture_output=True, text=True, check=True).stdout)
    source = validation["legacy_source_validation"]
    assert validation["protocol_valid"] is True
    assert source["legacy_source_root"].startswith("E:\\LJQ_code")
    assert [Path(row["path"]).name for row in source["required_files"]] == ["model.py"]
    assert "C:\\LJQ_code" not in source["legacy_source_root"]
    loaded = json.loads(subprocess.run([*common, "--self-test-frozen-models"], cwd=unrelated, env=environment, capture_output=True, text=True, check=True).stdout)
    assert loaded["frozen_model_self_test_valid"] is True
    assert [row["source_seed"] for row in loaded["models"]] == [3041, 3042, 3043, 3044, 3045]
    assert all(row["state_key_count"] > 0 and row["state_element_count"] > 0 for row in loaded["models"])
    assert loaded["data_opened"] is False and loaded["cuda_opened"] is False
