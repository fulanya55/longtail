from pathlib import Path


def test_same_pool_prefix_excludes_raw_and_rechunked_representations_only():
    root = Path("generated_data/p5g_shared_dense_execution_dev_seed3650000000_rechunked_v2")
    prefix = root.name.split("_rechunked", maxsplit=1)[0]
    assert "p5g_shared_dense_execution_dev_seed3650000000".startswith(prefix)
    assert "p5g_shared_dense_execution_dev_seed3660000000".startswith(prefix) is False
