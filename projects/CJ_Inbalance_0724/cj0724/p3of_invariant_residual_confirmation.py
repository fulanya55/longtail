"""Fixed P3o-E independent confirmation on one fresh identity pool."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cj0724.p3od_frozen_frequency_delay_parameter_residual as core
from cj0724.p3oe_invariant_frequency_delay_parameter_residual import InvariantFrozenFrequencyDelayResidual
from cj0724.p3n_cross_interference_backbone import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-seed", type=int, choices=(1130000000, 1140000000), required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    freeze = json.loads((project / "configs/p3of_invariant_residual_confirmation_method_freeze_20260802.json").read_text(encoding="utf-8"))
    if freeze["wrapper_sha256"] != sha256_file(Path(__file__).resolve()): raise RuntimeError("P3o-F wrapper hash mismatch")
    if freeze["implementation_sha256"] != sha256_file(Path(core.__file__).resolve()): raise RuntimeError("P3o-F core hash mismatch")
    core.FrozenFrequencyDelayResidual = InvariantFrozenFrequencyDelayResidual
    seed = args.confirmation_seed
    sys.argv = [sys.argv[0],
        "--train-root-template", "generated_data/p3oe_invariant_residual_train_seed1110000000_{regime}_rechunked",
        "--development-root-template", f"generated_data/p3oe_invariant_residual_confirmation_seed{seed}_{{regime}}_rechunked",
        "--preregister", "configs/p3of_invariant_residual_confirmation_preregister_20260802.json",
        "--method-freeze", "configs/p3of_invariant_residual_confirmation_method_freeze_20260802.json",
        "--output-root", f"outputs/p3of_invariant_residual_confirmation_seed{seed}_20260802",
        "--identity-seeds", "1110000000", str(seed)]
    core.main()


if __name__ == "__main__": main()
