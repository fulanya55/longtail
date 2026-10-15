import numpy as np
import pytest

from cjlt.data.parameter_schema import (
    MINIMAL_PARAMETER_SCHEMA,
    get_parameter_specs,
)
from cjlt.data.registry import (
    FOUR_CLASS_EXCLUDED_INDICES,
    FOUR_CLASS_INDICES,
    FOUR_CLASS_NAMES,
    SIX_CLASS_NAMES,
    get_class_names,
    get_selected_indices,
    select_labels,
    valid_sample_mask,
)


def test_class_registry_matches_0716_dataset():
    assert SIX_CLASS_NAMES == ["FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ"]
    assert FOUR_CLASS_NAMES == ["FMB", "SinFM", "MFT", "ISRJ"]
    assert FOUR_CLASS_INDICES == [0, 2, 3, 5]
    assert FOUR_CLASS_EXCLUDED_INDICES == [1, 4]
    assert get_class_names("four") == FOUR_CLASS_NAMES
    assert get_class_names("six") == SIX_CLASS_NAMES
    assert get_selected_indices("four") == FOUR_CLASS_INDICES
    assert get_selected_indices("six") == list(range(6))


def test_four_class_mask_excludes_hidden_sawfm_and_vft_components():
    labels6 = np.asarray(
        [
            [1, 0, 0, 0, 0, 0],  # keep: FMB
            [0, 0, 1, 1, 0, 1],  # keep: SinFM + MFT + ISRJ
            [1, 1, 0, 0, 0, 0],  # reject: contains SawFM
            [0, 0, 1, 0, 1, 0],  # reject: contains VFT
            [0, 0, 0, 0, 0, 0],  # reject: empty after selection
        ],
        dtype=np.uint8,
    )

    mask = valid_sample_mask(labels6, "four")

    assert mask.tolist() == [True, True, False, False, False]
    selected = select_labels(labels6[mask], "four")
    assert selected.shape == (2, 4)
    assert selected.tolist() == [[1, 0, 0, 0], [0, 1, 1, 1]]


def test_six_class_mask_keeps_nonempty_rows_and_rejects_empty_rows():
    labels6 = np.asarray(
        [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=np.uint8,
    )
    assert valid_sample_mask(labels6, "six").tolist() == [True, False]


def test_registry_rejects_unknown_mode_and_wrong_label_shape():
    with pytest.raises(ValueError, match="Unknown class mode"):
        get_class_names("seven")
    with pytest.raises(ValueError, match="shape"):
        valid_sample_mask(np.zeros((4, 4), dtype=np.uint8), "four")


def test_minimal_parameter_schema_uses_direct_nonredundant_slots():
    assert set(MINIMAL_PARAMETER_SCHEMA) == set(SIX_CLASS_NAMES)
    assert [(item.name, item.column_index) for item in get_parameter_specs("FMB")] == [
        ("JNR", 0),
        ("interference_bandwidth", 2),
    ]
    assert [(item.name, item.column_index) for item in get_parameter_specs("SawFM")] == [
        ("JNR", 11),
        ("sweep_period", 14),
    ]
    assert [(item.name, item.column_index) for item in get_parameter_specs("SinFM")] == [
        ("JNR", 16),
        ("frequency_deviation", 18),
    ]
    assert [(item.name, item.column_index) for item in get_parameter_specs("MFT")] == [
        ("JNR", 27),
        ("num_targets", 28),
    ]
    assert [(item.name, item.column_index) for item in get_parameter_specs("VFT")] == [
        ("JNR", 48),
        ("Doppler", 49),
    ]
    assert [(item.name, item.column_index) for item in get_parameter_specs("ISRJ")] == [
        ("JNR", 52),
        ("sample_width", 53),
    ]
    assert all(
        not item.derived
        for specs in MINIMAL_PARAMETER_SCHEMA.values()
        for item in specs
    )

