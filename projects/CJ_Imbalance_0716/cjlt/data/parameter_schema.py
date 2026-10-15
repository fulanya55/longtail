"""Minimal non-redundant parameter targets for the first experiment version."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    column_index: int
    unit: str
    normalization: str = "zscore"
    derived: bool = False


MINIMAL_PARAMETER_SCHEMA: Dict[str, List[ParameterSpec]] = {
    "FMB": [
        ParameterSpec("JNR", 0, "dB"),
        ParameterSpec("interference_bandwidth", 2, "Hz"),
    ],
    "SawFM": [
        ParameterSpec("JNR", 11, "dB"),
        ParameterSpec("sweep_period", 14, "s"),
    ],
    "SinFM": [
        ParameterSpec("JNR", 16, "dB"),
        ParameterSpec("frequency_deviation", 18, "Hz"),
    ],
    "MFT": [
        ParameterSpec("JNR", 27, "dB"),
        ParameterSpec("num_targets", 28, "count", normalization="minmax"),
    ],
    "VFT": [
        ParameterSpec("JNR", 48, "dB"),
        ParameterSpec("Doppler", 49, "Hz"),
    ],
    "ISRJ": [
        ParameterSpec("JNR", 52, "dB"),
        ParameterSpec("sample_width", 53, "s"),
    ],
}


def get_parameter_specs(class_name: str) -> List[ParameterSpec]:
    try:
        return list(MINIMAL_PARAMETER_SCHEMA[class_name])
    except KeyError:
        raise ValueError("Unknown class name: {!r}".format(class_name))

