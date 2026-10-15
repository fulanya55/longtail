"""Create visual checks for every combination in the 0716 closed-set train/test data.

Each output image contains two deterministic samples of one label combination:
the first and the middle record in its 400-sample group.  The four plots per
sample are real time-domain waveform, spectrum, LFM matched-filter output, and
time-frequency power.  The input MAT files are only read; no dataset content is
modified.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import fftconvolve, stft


CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
FS_HZ = 10e6
PULSE_WIDTH_S = 100e-6
LFM_BANDWIDTH_HZ = 5e6
SIGNAL_LENGTH = 3000
SAMPLES_PER_COMBINATION = 2
DB_FLOOR = -80.0


def db_normalized(values: np.ndarray) -> np.ndarray:
    """Return magnitude in dB relative to the largest value in this panel."""
    magnitude = np.abs(values)
    reference = max(float(np.max(magnitude)), np.finfo(float).tiny)
    return np.maximum(20.0 * np.log10(np.maximum(magnitude, np.finfo(float).tiny) / reference), DB_FLOOR)


def labels_to_name(label: tuple[int, ...]) -> str:
    return "+".join(name for name, active in zip(CLASS_NAMES, label) if active)


def safe_filename(value: str) -> str:
    return value.replace("+", "_")


def make_reference_chirp() -> np.ndarray:
    """Baseband 100-us, 5-MHz LFM pulse defined by the dataset documentation."""
    count = int(round(FS_HZ * PULSE_WIDTH_S))
    time_s = np.arange(count, dtype=np.float64) / FS_HZ
    slope_hz_s = LFM_BANDWIDTH_HZ / PULSE_WIDTH_S
    phase = np.pi * slope_hz_s * (time_s - PULSE_WIDTH_S / 2.0) ** 2
    return np.exp(1j * phase)


def plot_sample(axes: np.ndarray, iq: np.ndarray, sample_index: int, label_name: str, chirp: np.ndarray) -> None:
    signal = np.asarray(iq[0], dtype=np.float64) + 1j * np.asarray(iq[1], dtype=np.float64)
    time_us = np.arange(SIGNAL_LENGTH) / FS_HZ * 1e6

    axes[0].plot(time_us, signal.real, color="#175a9e", linewidth=0.55)
    axes[0].set_title("Real time-domain")
    axes[0].set_xlabel("Time (us)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    nfft = 8192
    frequency_mhz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / FS_HZ)) / 1e6
    spectrum = np.fft.fftshift(np.fft.fft(signal, n=nfft))
    axes[1].plot(frequency_mhz, db_normalized(spectrum), color="#8f2f2f", linewidth=0.6)
    axes[1].set_xlim(-FS_HZ / 2e6, FS_HZ / 2e6)
    axes[1].set_ylim(DB_FLOOR, 3)
    axes[1].set_title("Spectrum (relative)")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Magnitude (dB rel.)")
    axes[1].grid(alpha=0.25)

    compressed = fftconvolve(signal, np.conj(chirp[::-1]), mode="same")
    axes[2].plot(time_us, db_normalized(compressed), color="#347a34", linewidth=0.6)
    axes[2].set_ylim(DB_FLOOR, 3)
    axes[2].set_title("LFM matched-filter output")
    axes[2].set_xlabel("Range-time (us)")
    axes[2].set_ylabel("Magnitude (dB rel.)")
    axes[2].grid(alpha=0.25)

    frequency_hz, frame_s, zxx = stft(
        signal,
        fs=FS_HZ,
        window="hann",
        nperseg=256,
        noverlap=224,
        nfft=512,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    order = np.fft.fftshift(np.arange(frequency_hz.size))
    frequency_hz = np.fft.fftshift(frequency_hz)
    tf_db = db_normalized(zxx[order, :])
    image = axes[3].pcolormesh(
        frame_s * 1e6,
        frequency_hz / 1e6,
        tf_db,
        shading="auto",
        cmap="turbo",
        vmin=DB_FLOOR,
        vmax=0,
    )
    axes[3].set_ylim(-FS_HZ / 2e6, FS_HZ / 2e6)
    axes[3].set_title("Time-frequency power (relative)")
    axes[3].set_xlabel("Time (us)")
    axes[3].set_ylabel("Frequency (MHz)")
    colorbar = axes[3].figure.colorbar(image, ax=axes[3], pad=0.01)
    colorbar.set_label("dB rel.")

    axes[0].text(
        0.01,
        0.96,
        f"sample={sample_index}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    for axis in axes:
        axis.tick_params(labelsize=8)


def combination_groups(labels: np.ndarray) -> list[tuple[tuple[int, ...], list[int]]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(labels):
        groups[tuple(int(value) for value in row)].append(index)
    expected = (1 << len(CLASS_NAMES)) - 1
    if len(groups) != expected:
        raise ValueError(f"Expected {expected} nonempty combinations, found {len(groups)}")
    bad = {labels_to_name(key): len(value) for key, value in groups.items() if len(value) != 400}
    if bad:
        raise ValueError(f"Expected 400 samples per combination, found {bad}")
    return sorted(groups.items(), key=lambda item: (sum(item[0]), item[0]))


def read_iq(handle: h5py.File, sample_index: int) -> np.ndarray:
    raw = np.asarray(handle["data_legacy"][:, sample_index], dtype=np.float64)
    if raw.shape != (2 * SIGNAL_LENGTH,):
        raise ValueError(f"Unexpected sample shape {raw.shape} at index {sample_index}")
    return raw.reshape(2, SIGNAL_LENGTH)


def process_split(mat_path: Path, output_root: Path) -> None:
    split_name = mat_path.stem.replace("closedset_6class_", "")
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, object]] = []
    chirp = make_reference_chirp()

    with h5py.File(mat_path, "r") as handle:
        labels = np.asarray(handle["label_selected"][:, :], dtype=np.uint8).T
        groups = combination_groups(labels)
        pdf_path = split_dir / f"{split_name}_all_combinations.pdf"
        with PdfPages(pdf_path) as pdf:
            for order, (label, indices) in enumerate(groups, start=1):
                label_name = labels_to_name(label)
                chosen_positions = np.linspace(0, len(indices) - 1, SAMPLES_PER_COMBINATION, dtype=int)
                chosen_indices = [indices[position] for position in chosen_positions]
                figure, axes = plt.subplots(
                    nrows=SAMPLES_PER_COMBINATION,
                    ncols=4,
                    figsize=(16, 6.1 * SAMPLES_PER_COMBINATION),
                    constrained_layout=True,
                )
                figure.suptitle(
                    f"{split_name.upper()} | combination {order:02d}/63: {label_name} | "
                    f"FS=10 MHz, LFM=5 MHz x 100 us",
                    fontsize=14,
                    fontweight="bold",
                )
                for row, sample_index in enumerate(chosen_indices):
                    plot_sample(axes[row], read_iq(handle, sample_index), sample_index, label_name, chirp)
                    selection_rows.append(
                        {
                            "split": split_name,
                            "combination_order": order,
                            "combination": label_name,
                            "sample_index": sample_index,
                            "selection_position_within_combination": int(chosen_positions[row]),
                        }
                    )
                figure_path = split_dir / f"{order:02d}_{safe_filename(label_name)}.png"
                figure.savefig(figure_path, dpi=150, bbox_inches="tight")
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
                print(f"[{split_name}] {order:02d}/63 {label_name}: {chosen_indices}", flush=True)

    with (split_dir / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)


def write_readme(output_root: Path) -> None:
    text = """# 0716 closed-set train/test visual inspection\n\nThis folder is generated from the two root datasets only:\n\n- `closedset_6class_train.mat`\n- `closedset_6class_test.mat`\n\nFor every one of the 63 nonempty label combinations, two deterministic records are shown:\nthe first and the middle record in the combination's 400 samples. Each image contains\nreal time-domain waveform, complex-signal spectrum, 5 MHz x 100 us LFM matched-filter\noutput, and complex-signal STFT. Spectral, matched-filter and STFT panels are normalized\nindependently to their own peak (dB relative); time-domain amplitude is not normalized.\n\n`train_all_combinations.pdf` and `test_all_combinations.pdf` give a paged overview.\nThe PNG files are one combination per image, and `selection.csv` records the selected\nindices. The original MAT files are not changed.\n"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\\LJQ_code\\干扰数据\\复合干扰数据0716"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\\LJQ_code\\干扰数据\\复合干扰数据0716\\visual_check_train_test_20260724"),
    )
    args = parser.parse_args()
    files = (
        args.data_root / "closedset_6class_train.mat",
        args.data_root / "closedset_6class_test.mat",
    )
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_root)
    for path in files:
        process_split(path, args.output_root)


if __name__ == "__main__":
    main()
