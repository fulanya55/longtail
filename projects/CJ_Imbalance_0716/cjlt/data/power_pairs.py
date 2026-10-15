"""Identity-safe pairing across the nine power-sweep training levels."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np


PathLike = Union[str, Path]
FIXED_LABEL_INDICES = (0, 1, 2)
SWEPT_LABEL_INDICES = (3, 4, 5)
FIXED_JNR_DB = 30.0


@dataclass
class PowerPairRecord:
    identity: int
    reference_level: int
    paired_level: int
    reference_index: int
    paired_index: int
    labels: np.ndarray
    reference_effective_jnr: np.ndarray
    paired_effective_jnr: np.ndarray
    stronger_level_by_label: np.ndarray
    relative_power_gap: float


@dataclass
class PowerPairIndex:
    reference_level: int
    levels: Tuple[int, ...]
    records: List[PowerPairRecord]
    sha256: str


@dataclass
class RoleBoundPowerPairIndex:
    """Power pairs whose identities are inherited from a base split role."""

    base_indices: Tuple[int, ...]
    reference_indices: Tuple[int, ...]
    reference_level: int
    levels: Tuple[int, ...]
    records: List[PowerPairRecord]
    sha256: str


def discover_power_sweep_train_files(root: PathLike) -> Dict[int, Path]:
    root_path = Path(root)
    manifest_path = root_path / "power_sweep_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Power-sweep manifest not found: {}".format(manifest_path))
    files = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            level = int(float(row["sweep_JNR_dB"]))
            train_path = Path(row["train_path"])
            if not train_path.is_file():
                # Manifests are moved with the dataset between workstations.
                # Rebase stale absolute paths to the manifest's current root.
                output_name = Path(row.get("output_dir", "")).name
                candidates = []
                if output_name:
                    candidates.append(root_path / output_name / train_path.name)
                candidates.extend(
                    sorted(root_path.glob("level_*__jnr_{}dB/{}".format(level, train_path.name)))
                )
                train_path = next((path for path in candidates if path.is_file()), train_path)
            if not train_path.is_file():
                raise FileNotFoundError(
                    "Power-sweep train file not found for {} dB: {}".format(
                        level, train_path
                    )
                )
            files[level] = train_path
    if not files:
        raise ValueError("Power-sweep manifest contains no train files")
    return {level: files[level] for level in sorted(files)}


def load_level_identity(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    mat_path = Path(path)
    with h5py.File(str(mat_path), "r") as handle:
        seeds = np.asarray(handle["receiver_noise_seed"][:, :]).reshape(-1)
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
    rounded = np.rint(seeds).astype(np.int64)
    if not np.allclose(seeds, rounded.astype(np.float64), rtol=0.0, atol=0.0):
        raise ValueError("receiver_noise_seed contains non-integral values")
    if labels.shape != (len(rounded), 6):
        raise ValueError(
            "Expected labels [N, 6] aligned with seeds, got {}".format(labels.shape)
        )
    if len(np.unique(rounded)) != len(rounded):
        raise ValueError("receiver_noise_seed must be unique within each level")
    return rounded, np.ascontiguousarray(labels)


def _effective_jnr(labels: np.ndarray, level: int) -> np.ndarray:
    values = np.full(6, np.nan, dtype=np.float32)
    for index in FIXED_LABEL_INDICES:
        if labels[index]:
            values[index] = FIXED_JNR_DB
    for index in SWEPT_LABEL_INDICES:
        if labels[index]:
            values[index] = float(level)
    return values


def _relative_strength_scores(labels: np.ndarray, level: int) -> np.ndarray:
    scores = np.full(6, np.nan, dtype=np.float32)
    fixed_present = bool(np.any(labels[list(FIXED_LABEL_INDICES)]))
    swept_present = bool(np.any(labels[list(SWEPT_LABEL_INDICES)]))
    for index in FIXED_LABEL_INDICES:
        if labels[index]:
            scores[index] = (
                FIXED_JNR_DB - float(level) if swept_present else FIXED_JNR_DB
            )
    for index in SWEPT_LABEL_INDICES:
        if labels[index]:
            scores[index] = (
                float(level) - FIXED_JNR_DB if fixed_present else float(level)
            )
    return scores


def _stronger_level_by_label(
    labels: np.ndarray, reference_level: int, paired_level: int
) -> np.ndarray:
    reference_scores = _relative_strength_scores(labels, reference_level)
    paired_scores = _relative_strength_scores(labels, paired_level)
    stronger = np.zeros(6, dtype=np.int16)
    for index in range(6):
        if not labels[index]:
            continue
        delta = float(paired_scores[index] - reference_scores[index])
        if delta > 1e-9:
            stronger[index] = int(paired_level)
        elif delta < -1e-9:
            stronger[index] = int(reference_level)
    return stronger


def _hash_index(
    reference_level: int,
    levels: Sequence[int],
    records: Sequence[PowerPairRecord],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(reference_level)).encode("ascii"))
    digest.update(",".join(str(int(level)) for level in levels).encode("ascii"))
    for record in records:
        digest.update(
            np.asarray(
                [
                    record.identity,
                    record.reference_level,
                    record.paired_level,
                    record.reference_index,
                    record.paired_index,
                ],
                dtype="<i8",
            ).tobytes()
        )
        digest.update(np.asarray(record.labels, dtype=np.uint8).tobytes())
        digest.update(
            np.asarray(record.stronger_level_by_label, dtype="<i2").tobytes()
        )
    return digest.hexdigest()


def build_power_pair_index(
    root: PathLike,
    levels: Optional[Sequence[int]] = None,
    reference_level: int = 30,
    sample_indices: Optional[Iterable[int]] = None,
) -> PowerPairIndex:
    files = discover_power_sweep_train_files(root)
    selected_levels = tuple(
        sorted(files) if levels is None else sorted(int(value) for value in levels)
    )
    if int(reference_level) not in selected_levels:
        raise ValueError("reference_level must be included in levels")
    missing = [level for level in selected_levels if level not in files]
    if missing:
        raise ValueError("Requested power levels are missing: {}".format(missing))

    reference_seeds, reference_labels = load_level_identity(files[int(reference_level)])
    if sample_indices is None:
        reference_indices = np.arange(len(reference_seeds), dtype=np.int64)
    else:
        reference_indices = np.asarray(list(sample_indices), dtype=np.int64)
    if reference_indices.ndim != 1:
        raise ValueError("sample_indices must be one-dimensional")
    if reference_indices.size and (
        reference_indices.min() < 0
        or reference_indices.max() >= len(reference_seeds)
    ):
        raise IndexError("sample_indices contain an out-of-range index")

    records = []
    for paired_level in selected_levels:
        if paired_level == int(reference_level):
            continue
        paired_seeds, paired_labels = load_level_identity(files[paired_level])
        paired_lookup = {
            int(identity): int(index) for index, identity in enumerate(paired_seeds)
        }
        for reference_index in reference_indices:
            ref_index = int(reference_index)
            identity = int(reference_seeds[ref_index])
            if identity not in paired_lookup:
                raise ValueError(
                    "Identity {} missing from {} dB level".format(
                        identity, paired_level
                    )
                )
            paired_index = paired_lookup[identity]
            labels = reference_labels[ref_index].copy()
            if not np.array_equal(labels, paired_labels[paired_index]):
                raise ValueError(
                    "Label mismatch for identity {} between {} and {} dB".format(
                        identity, reference_level, paired_level
                    )
                )
            records.append(
                PowerPairRecord(
                    identity=identity,
                    reference_level=int(reference_level),
                    paired_level=int(paired_level),
                    reference_index=ref_index,
                    paired_index=int(paired_index),
                    labels=labels,
                    reference_effective_jnr=_effective_jnr(
                        labels, int(reference_level)
                    ),
                    paired_effective_jnr=_effective_jnr(labels, int(paired_level)),
                    stronger_level_by_label=_stronger_level_by_label(
                        labels, int(reference_level), int(paired_level)
                    ),
                    relative_power_gap=float(
                        abs(int(paired_level) - int(reference_level))
                    ),
                )
            )

    return PowerPairIndex(
        reference_level=int(reference_level),
        levels=selected_levels,
        records=records,
        sha256=_hash_index(int(reference_level), selected_levels, records),
    )


def build_role_bound_power_pair_index(
    base_path: PathLike,
    root: PathLike,
    base_indices: Iterable[int],
    levels: Optional[Sequence[int]] = None,
    reference_level: int = 30,
) -> RoleBoundPowerPairIndex:
    """Map a pre-existing base role to power views using receiver identity.

    The base MAT and power-sweep MAT files contain the same simulated identities.
    Mapping by ``receiver_noise_seed`` prevents independent re-splitting and leakage.
    """

    base_seed, _ = load_level_identity(base_path)
    selected_base = np.asarray(list(base_indices), dtype=np.int64)
    if selected_base.ndim != 1:
        raise ValueError("base_indices must be one-dimensional")
    if selected_base.size and (
        selected_base.min() < 0 or selected_base.max() >= len(base_seed)
    ):
        raise IndexError("base_indices contain an out-of-range index")

    files = discover_power_sweep_train_files(root)
    if int(reference_level) not in files:
        raise ValueError("reference power level is unavailable")
    reference_seed, _ = load_level_identity(files[int(reference_level)])
    reference_lookup = {
        int(identity): int(index) for index, identity in enumerate(reference_seed)
    }
    reference_indices = []
    for base_index in selected_base:
        identity = int(base_seed[int(base_index)])
        if identity not in reference_lookup:
            raise ValueError(
                "Base identity {} is missing from {} dB reference data".format(
                    identity, reference_level
                )
            )
        reference_indices.append(reference_lookup[identity])

    paired = build_power_pair_index(
        root=root,
        levels=levels,
        reference_level=int(reference_level),
        sample_indices=reference_indices,
    )
    digest = hashlib.sha256()
    digest.update(paired.sha256.encode("ascii"))
    digest.update(np.asarray(selected_base, dtype="<i8").tobytes())
    digest.update(np.asarray(reference_indices, dtype="<i8").tobytes())
    return RoleBoundPowerPairIndex(
        base_indices=tuple(int(value) for value in selected_base),
        reference_indices=tuple(int(value) for value in reference_indices),
        reference_level=paired.reference_level,
        levels=paired.levels,
        records=paired.records,
        sha256=digest.hexdigest(),
    )
