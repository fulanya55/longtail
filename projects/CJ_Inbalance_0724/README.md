# CJ_Inbalance_0724: I/Q interface reproduction base

This is a small, isolated six-label compound-jamming training base built for
comparing I/Q computation interfaces on the 0716 data.  It deliberately does
not copy the previous long-tail, parameter-regression, pseudo-label, or
power-view pipelines.

The EfficientNet-B1 frequency--visibility dual-axis development experiment is
now implemented and complete: 33 formal runs, identity-safe nine-level power
views, exact IF=3 manifests, BCE/DB and single/paired-view ablations. The
Chinese result report is `docs/efficientnet_b1_dual_axis_results_20260724.md`;
the frozen test set was not used.

The follow-up VFT-D11 hard-negative screen is also complete. OHEM, OHEM plus
ranking, and a bounded specialized residual expert were compared against the
frozen EfficientNet-B1 plus BCE baseline with three matched seeds. None passed
the promotion gate; see
`docs/efficientnet_b1_vft_hard_negative_results_20260725.md`. The locked test
set was not used.

The next VFT study tested online/frozen positive-reference distillation and a
deployable Doppler-matched-filter branch. Both distillation mechanisms failed
their one-seed stop gate. The physics branch passed the three-seed development
gate, raising mean VFT AUPR from 0.5929 to 0.9260 while reducing VFT FPR from
0.4559 to 0.1516. See
`docs/efficientnet_b1_vft_physics_fusion_results_20260725.md`. This is still
development evidence; the locked test remains unused.

E5 has now also passed a preregistered confirmation on two untouched seeds
(`271828`, `314159`). Across all five development seeds, VFT AUPR is
`0.9253 +/- 0.0048`, with a mean improvement of `+0.3372`; every seed selected
physics weight 1.0. The frozen candidate bundle and hashes are under
`outputs/vft_physics_e5_frozen_candidate_20260725/`; see
`docs/e5_confirmation_and_frozen_candidate_20260725.md`. Locked-test access is
still not authorized.

## Fixed data contract

- data: `E:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat`
- signal: `data_legacy[:, sample]`, reshaped to `[2, 3000]` as `[I, Q]`
- labels: `label_selected[:, sample]`, transposed to six-label `[6]`
- label order: `FMB, SawFM, SinFM, MFT, VFT, ISRJ`
- normalization: complex RMS per sample, preserving phase and I/Q geometry

The frozen `closedset_6class_test.mat` is never read by the default scripts.
The training file is split deterministically into train and validation sets and
the resulting indices are saved with every run.

## Compared interfaces

| CLI name | Computation | What is controlled |
| --- | --- | --- |
| `complex1d` | true complex 1D convolution: `Wr*I - Wi*Q`, `Wr*Q + Wi*I` | explicit complex multiplication and magnitude fusion |
| `iq_grouped` | I/Q-separated grouped 1D convolutions followed by a 1x1 fusion | two independent I/Q filter banks before learned cross-IQ mixing |

Both models receive the same RMS-normalized `[B, 2, 3000]` tensors, use the
same multilabel `BCEWithLogitsLoss`, optimizer, split seed, and threshold.

## Standard modern-convolution baselines

These are native implementations in `cj0724/models.py`, not image-pretrained
models. They receive the current I/Q tensor directly as two real-valued input
channels and are trained from scratch.

| CLI name | Reproduced architecture | 1D/I-Q adaptation |
| --- | --- | --- |
| `efficientnet_b0_1d` | EfficientNet-B0: standard MBConv stage schedule, depthwise convolution, SE, SiLU, stochastic depth | stem `Conv1d(2,32)`; original B0 widths and repeat counts |
| `efficientnet_b1_1d` | EfficientNet-B1 | fixed 3,000-sample input; B1 depth coefficient 1.1. Image-resolution scaling has no meaningful direct 1D equivalent. |
| `convnext_tiny_1d` | ConvNeXt-Tiny: 4 stages with depths `3,3,9,3`, dimensions `96,192,384,768` | `Conv1d(2,96,k=4,s=4)` patchify stem; 7-tap depthwise Conv1d + LayerNorm + 4x channel MLP blocks |

`efficientnet_b0_1d` and `efficientnet_b1_1d` are the appropriate standard
EfficientNet baselines. The older `legacy_effnet_v1/v2/v3` entries below are
your earlier I/Q-specific variants and should not be described as canonical
EfficientNet B0/B1.

## Legacy-model compatibility set

The following definitions are reused from
`C:\LJQ_code\pythonProject\新建文件夹` through a read-only compatibility
adapter; their original source files remain the single source of truth.

| CLI name | Original source | I/Q adapter |
| --- | --- | --- |
| `legacy_mlcnn` | `model.py: cnnnet` | I/Q concatenate to legacy `[B,1,6000]` |
| `legacy_resnet34` | `DBmodel.py: ResNet34_1D_Complex` | I/Q concatenate to legacy `[B,1,6000]` |
| `legacy_shufflenetv2` | `DBmodel.py: ShuffleNetV2_1D_Complex` | I/Q concatenate to legacy `[B,1,6000]` |
| `legacy_mobilenetv2` | `DBmodel.py: MobileNetV2_1D_Complex` | I/Q concatenate to legacy `[B,1,6000]` |
| `legacy_effnet_v1` | `EffNetV1.py: EfficientNet` | I/Q concatenate to legacy `[B,1,6000]` |
| `legacy_effnet_v2` | `EffNetV2.py: EfficientNet` | I/Q concatenate to legacy `[B,1,6000]` |
| `legacy_effnet_v3` | `EffNetV3.py: EfficientNetV3` | I/Q concatenate; choose V3 attention with `--legacy-attention` |
| `legacy_effnet_2d_iqm` | `EffNet.py: EfficientNet` | construct legacy three-channel 2D input `[I,Q,|z|]` with height 1 |

The adapters make all models output six raw logits. They do not load old
weights, because the earlier four- and six-class checkpoints have incompatible
experimental provenance. All compatibility models therefore train from scratch.

## Environment

The verified local environment is `PY10` (`torch 2.11.0+cu128`, `h5py 3.10`).
The base Python environment currently has an incompatible `h5py`/NumPy pair,
so use `conda run -n PY10` or activate `PY10`.

## Commands

Inspect the actual MAT-v7.3 top-level layout:

```powershell
conda run -n PY10 python -m cj0724.inspect_data
```

Run a fast real-data smoke check (8 samples per model, one backward pass):

```powershell
conda run -n PY10 python scripts\smoke_real_data.py
```

Start a development comparison (the defaults use all 25,200 training samples):

```powershell
conda run -n PY10 python -m cj0724.train --model complex1d --epochs 30 --seed 20260724
conda run -n PY10 python -m cj0724.train --model iq_grouped --epochs 30 --seed 20260724
conda run -n PY10 python -m cj0724.train --model efficientnet_b0_1d --epochs 30 --seed 20260724
conda run -n PY10 python -m cj0724.train --model efficientnet_b1_1d --epochs 30 --seed 20260724
conda run -n PY10 python -m cj0724.train --model convnext_tiny_1d --epochs 30 --seed 20260724
conda run -n PY10 python -m cj0724.train --model legacy_resnet34 --epochs 30 --seed 20260724
conda run -n PY10 python -m cj0724.train --model legacy_effnet_v3 --legacy-attention eca --epochs 30 --seed 20260724
```

For a short code-path run, explicitly cap the sample counts:

```powershell
conda run -n PY10 python -m cj0724.train --model complex1d --epochs 1 --max-train-samples 256 --max-val-samples 64
```

Outputs are written under `outputs/<model>_seed<seed>_<timestamp>/` and include
the model configuration, split indices, epoch metrics, and the best validation
checkpoint.

## Fair backbone benchmark

Use `cj0724.benchmark` when the research question is the backbone itself. It
removes every model-specific classifier and applies the same pipeline:

```text
raw backbone feature
 -> Linear(raw_dim, 256) + LayerNorm + GELU
 -> identical Dropout(0.2) + Linear(256, 6) classifier
 -> BCEWithLogitsLoss
```

The benchmark also fixes the data split, random seed, shuffled sample order,
normalization, optimizer, learning rate, weight decay, batch size, epoch budget,
threshold, and checkpoint rule. The default checkpoint rule is validation exact
subset accuracy first, macro-F1 second, and validation loss third. The locked
test file is not read.

Run all registered backbones with one controlled recipe:

```powershell
conda run -n PY10 python -m cj0724.benchmark `
  --epochs 30 `
  --batch-size 64 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --seed 20260724
```

The shared split, protocol, per-model histories/checkpoints, and `summary.csv`
are written under `outputs/backbone_benchmark/seed<seed>_<timestamp>/`.

After the checkpoints are available, detailed evaluation produces:

- `per_label_metrics.csv`: TP/TN/FP/FN, accuracy, precision, recall,
  specificity, and F1 for FMB/SawFM/SinFM/MFT/VFT/ISRJ;
- `per_combination_metrics.csv`: support, exact combination accuracy, mean
  label accuracy, and predicted cardinality for all 63 nonempty combinations;
- `per_label_accuracy_comparison.csv` and `per_label_f1_comparison.csv`: wide
  tables comparing all backbones label by label;
- `per_combination_accuracy_comparison.csv`: one row per combination and one
  column per backbone;
- `detailed_summary.csv`: macro label accuracy/F1, macro combination accuracy,
  and worst-combination accuracy for every backbone.

The benchmark runs this evaluation automatically when all models finish. It can
also be rerun from saved checkpoints without retraining:

```powershell
conda run -n PY10 python -m cj0724.evaluate_benchmark `
  --run-dir outputs\backbone_benchmark\seed20260724_20260724_090209
```

## Scientific boundary

This is an interface/architecture reproduction base, not a claimed long-tail
protocol. The root 0716 train data are balanced. Do not use the nine-level
power-sweep views as independent rows or mix source identities across splits;
add a separate identity-safe split builder before studying power or visibility.

## VFT-D11 multi-expert and long-tail-loss screen

The controlled 2026-07-25 screen adds a shared-trunk three-expert
EfficientNet-B1 and asymmetric loss without changing the identity-safe dual-axis
protocol. Reproduce the 12 formal runs with:

```powershell
conda run -n PY10 python -m cj0724.improvement_matrix `
  --output-root outputs/vft_d11_backbone_loss_20260725 `
  --epochs 30

conda run -n PY10 python -m cj0724.aggregate_improvements `
  --run-root outputs/vft_d11_backbone_loss_20260725

conda run -n PY10 python -m cj0724.diagnose_improvements `
  --run-root outputs/vft_d11_backbone_loss_20260725
```

The result is a negative promotion decision for default ASL and naive
equal-task experts. The detailed evidence and next specialized-expert direction
are recorded in `docs/efficientnet_b1_moe_longtail_screen_20260725.md`.
