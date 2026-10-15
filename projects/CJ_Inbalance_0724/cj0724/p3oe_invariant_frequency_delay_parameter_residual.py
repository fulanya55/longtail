"""P3o-E: invariant frozen-frequency delay parameter residual on fresh identities."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
import cj0724.p3od_frozen_frequency_delay_parameter_residual as core
from cj0724.p3n_cross_interference_backbone import sha256_file


class InvariantFrozenFrequencyDelayResidual(core.FrozenFrequencyDelayResidual):
    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self


def main():
    project = Path(__file__).resolve().parents[1]
    freeze = json.loads((project / "configs/p3oe_invariant_frequency_delay_parameter_residual_method_freeze_20260802.json").read_text(encoding="utf-8"))
    if freeze["wrapper_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P3o-E wrapper hash mismatch")
    if freeze["implementation_sha256"] != sha256_file(Path(core.__file__).resolve()):
        raise RuntimeError("P3o-E core hash mismatch")
    core.FrozenFrequencyDelayResidual = InvariantFrozenFrequencyDelayResidual
    sys.argv.extend([
        "--train-root-template", "generated_data/p3oe_invariant_residual_train_seed1110000000_{regime}_rechunked",
        "--development-root-template", "generated_data/p3oe_invariant_residual_dev_seed1120000000_{regime}_rechunked",
        "--preregister", "configs/p3oe_invariant_frequency_delay_parameter_residual_preregister_20260802.json",
        "--method-freeze", "configs/p3oe_invariant_frequency_delay_parameter_residual_method_freeze_20260802.json",
        "--output-root", "outputs/p3oe_invariant_frequency_delay_parameter_residual_development_20260802",
        "--identity-seeds", "1110000000", "1120000000",
    ])
    core.main()


if __name__ == "__main__":
    main()
