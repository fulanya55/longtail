from pathlib import Path

from cj0724.p3om_data_audit import role_paths


def test_p3om_role_paths_keep_explicit_rechunk_suffix():
    paths = role_paths(Path("D:/CJ0724_rechunked_data"), "train", 1570000000, "_rechunked_v2")
    assert set(paths) == {"ID_control", "low_count", "close_spacing", "ultra_close", "weak_target"}
    assert all("_rechunked_v2" in str(path) for files in paths.values() for path in files.values())
