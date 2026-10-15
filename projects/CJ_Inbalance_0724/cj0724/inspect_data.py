"""Inspect only the fields used by this base, without walking MATLAB reference groups."""

from __future__ import annotations

import argparse

import h5py

from cj0724 import DEFAULT_TRAIN_MAT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default=DEFAULT_TRAIN_MAT)
    args = parser.parse_args()
    wanted = ("data_legacy", "iq_complex", "label_selected", "parameter_values", "parameter_mask", "receiver_noise_seed")
    with h5py.File(args.data_file, "r") as handle:
        for key in wanted:
            if key not in handle:
                print(f"{key}: MISSING")
                continue
            dataset = handle[key]
            print(f"{key}: shape={tuple(dataset.shape)}, dtype={dataset.dtype}")


if __name__ == "__main__":
    main()
