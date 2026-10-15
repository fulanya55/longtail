# Route B literature-to-experiment plan

## 1. Research question

Under the fixed `[N,2,3000]` I/Q contract and an identity-isolated
development-v2/test-v2 protocol, can conditional parameter heads improve
12-parameter estimation for FMB, SawFM, SinFM, MFT, VFT and ISRJ without
changing the six-class recognition output? The experimental claims below are
proposals until the preregistered development and independent test roles are
complete.

## 2. Literature audit and evidence tier

`[F]` means the full article or publisher full text was inspected; `[A]` means
the abstract/metadata was inspected; `[T]` means title/metadata only. DOI links
are retained so an exact version can be re-audited.

| Area | Article and year | Object/problem | Model or parameter mechanism | Experimental cue for Route B | Tier |
|---|---|---|---|---|:---:|
| Communication | *Deep learning based OFDM parameter estimation*, IEEE Access, 2021, [DOI](https://doi.org/10.1109/ACCESS.2021.3058732) | Estimate OFDM waveform parameters under channel/noise variation | CNN-style learned feature mapping for modulation/waveform parameters | Keep waveform input fixed; report parameter MAE by SNR and class | [A] |
| Communication | *Deep learning based phase parameter estimation*, IEEE Communications Letters, 2021, [DOI](https://doi.org/10.1109/LCOMM.2021.3061914) | Estimate phase-related signal parameters | CNN plus recurrent temporal aggregation | Motivates temporal auxiliary branch, but not a reason to change the recognition head | [A] |
| Communication | *SigdetNet: A deep neural network for signal detection and parameter estimation*, Sensors, 2022, [article](https://www.mdpi.com/1424-8220/22/7/2559) | Joint signal detection and parameter estimation in noisy observations | Shared learned representation with detection/estimation outputs | Separately report detection safety and masked parameter loss | [F] |
| Radar | *Model-based time-series clustering for radar pulse deinterleaving*, IEEE TAES, 2021, [DOI](https://doi.org/10.1109/TAES.2021.3053475) | Group pulses from multiple emitters | Model-based temporal similarity/clustering | Source-identity grouping is more important than random sample splitting | [A] |
| Radar | *Neural machine translation for radar pulse deinterleaving*, IEEE TAES, 2022, [DOI](https://doi.org/10.1109/TAES.2022.3142642) | Sequence-level deinterleaving of interleaved pulse streams | Encoder-decoder sequence model | Parameter/recognition claims need source-identity and sequence-role manifests | [A] |
| Radar | *Deep learning-based radar pulse deinterleaving via semantic segmentation*, IEEE TSP, 2022, [DOI](https://doi.org/10.1109/TSP.2022.3177944) | Label pulse ownership in a stream | Semantic segmentation over pulse-time structure | Supports an auxiliary temporal/spectral branch, not pseudo-labeling | [A] |
| Radar | *Radar signal deinterleaving based on pulse description words and PRI*, EURASIP JASP, 2021, [DOI](https://doi.org/10.1186/s13634-021-00772-0) | Exploit PDW/PRI structure | Handcrafted PRI/PDW sequence features | Use physical parameters only as observable auxiliary inputs; no label leakage | [F] |
| Radar | *Disentangled representation learning for radar signal recognition*, Remote Sensing, 2024, [article](https://www.mdpi.com/2072-4292/16/4/677) | Separate nuisance and signal factors | Disentangled latent representation | Motivation for ablation of shared versus task-specific parameter features | [F] |
| Emitter | *Emitter identification based on parameter clustering*, Remote Sensing, 2022, [article](https://www.mdpi.com/2072-4292/14/20/5152) | Cluster emitters from measured signal parameters | Parameter clustering and identity-level grouping | Confirms that source identity must be retained in split manifests | [F] |
| Emitter | *A CNN-based radar emitter identification method*, IET Radar, Sonar & Navigation, 2020, [DOI](https://doi.org/10.1049/iet-rsn.2019.0448) | Identify emitters from raw/derived signal representations | CNN classifier | Recognition baseline is a control, not evidence for parameter regression | [A] |
| Emitter | *Few-shot radar emitter identification based on a self-supervised feature adaptation network*, IET Radar, Sonar & Navigation, 2023, [DOI](https://doi.org/10.1049/rsn2.12357) | Identification with few labeled examples | Self-supervised feature adaptation | Relevant to long-tail/open-world proposal only; not mixed into R1 | [A] |
| Emitter | *Open-set radar emitter identification based on MSAK-CNN-LSTM*, IET Radar, Sonar & Navigation, 2024, [DOI](https://doi.org/10.1049/rsn2.12550) | Open-set emitter recognition | Multi-scale attention CNN plus LSTM | Open-world extension is a separate proposal and cannot use locked-test feedback | [A] |
| Emitter | *Hierarchical Bayesian nonparametric modelling for radar emitter recognition*, IEEE TPS, 2025, [DOI](https://doi.org/10.1109/TPS.2025.3521690) | Discover/recognize emitters under changing populations | Hierarchical Bayesian nonparametric model with temporal features | Supports future identity-isolated grouping; not a substitute for current six-class R1 | [A] |
| Jamming | *DAT-Net: Deep adversarial training network for radar jamming recognition*, EURASIP JASP, 2023, [article](https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13634-023-00969-1) | Recognize jamming under adversarial/noisy perturbation | Deep adversarial training and CNN feature learning | Robustness is a post-R1 stress test; do not change the frozen R1 budget | [F] |
| Jamming | *Deep neural network based recognition of interrupted-sampling repeater jamming*, IEEE JSTARS, 2022, [DOI](https://doi.org/10.1109/JSTARS.2022.3141929) | Recognize ISRJ and estimate related structure | Deep network over signal/time-frequency features | ISRJ JNR and sample-width are explicit masked targets in the 12-parameter schema | [A] |
| Jamming | *Recognition of compound interrupted-sampling repeater jamming*, IEEE SPL, 2024, [DOI](https://doi.org/10.1109/LSP.2024.3383965) | Compound ISRJ recognition | Multi-component learned representation | Directly motivates compound-label and conditional-head controls | [A] |
| Jamming | *YOLOv8-OBB based radar jamming signal recognition*, Remote Sensing, 2025, [article](https://www.mdpi.com/2072-4292/17/4/657) | Detect jamming events in time-frequency images | Oriented-object detector | Detection framing is an optional auxiliary branch, not the fixed I/Q input replacement | [F] |
| Jamming | *Variable-length deception jamming recognition*, Sensors, 2025, [article](https://www.mdpi.com/1424-8220/25/4/1103) | Recognize deception jamming with variable observation lengths | Variable-length sequence model | Current Route B keeps length fixed; variable-length is a separate proposal | [F] |
| Jamming | *Radar jamming recognition with YOLOv8-SimAM*, 2025, [DOI](https://doi.org/10.1109/ICSPCC63362.2025.11127935) | Time-frequency image detection/recognition | YOLOv8 with attention module | Attention is not counted as an innovation without a matched ablation | [A] |

### Literature-derived design constraints

1. A shared representation is a useful control, but parameter gradients can
   damage recognition; therefore B1/B2/B3 must be paired by seed and share all
   input, split, optimizer, loss and budget choices.
2. PSD/STFT features are auxiliary observables. They must be generated from the
   same I/Q sample, with the recognition head unchanged and no parameter target,
   label or test feedback entering the feature path.
3. Physical parameter validity is conditional. Inactive NaNs are not values;
   they are excluded by `label=1 AND parameter_mask=1 AND finite target`.
4. Emitter/source literature makes identity grouping a validity condition. The
   nine power views are paired views sharing `receiver_noise_seed`, not nine
   independent samples.

## 3. Route B model definitions

### R1 fixed baselines

- **B0**: legacy MLCNN classification-only.
- **B1**: legacy MLCNN shared backbone plus one global parameter head.
- **B2**: shared backbone plus six class-conditioned parameter subheads.
- **B3**: B2 with parameter-input stop-gradient.
- **B4**: B2 with two-epoch parameter-loss warm-up.

### R2 proposal branches

- **Physical-FiLM**: 26-dimensional fixed time/spectrum/ACF descriptors from
  I/Q condition only the parameter representation; recognition logits are
  detached before conditioning.
- **Spectral-FiLM**: 19-dimensional normalized positive-frequency PSD descriptor
  (16 bands, centroid, entropy, peak) condition only the parameter
  representation. The six-class recognition head and input shape are unchanged.

Both branches are implemented and smoke-tested, but neither has a performance
claim until a matched multi-seed development comparison is complete.

## 4. Fixed training and evaluation protocol

- Active data: `E:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat`.
- Input: `[N,2,3000]`; class order `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]`.
- Development split: combination-stratified, seed `20260826`, 90/10, with
  train/validation index hashes recorded in the preregistration.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Parameter loss: masked Smooth-L1, weight `0.1`, with B4 warm-up fixed before
  training.
- Threshold: `0.5`; no test-v2 or historical locked test is opened for R1/R2/R3.
- Seeds: `2041,2042,2043,2044,2045`.
- Primary metrics: classification Macro-AUPR/Macro-F1 and 12-parameter
  Macro-NMAE; secondary metrics are per-class, per-parameter, visibility and
  paired-power tables.

## 5. Hard gates and stopping rules

The current numerical gates are explicitly marked as a proposal pending user
confirmation: each conditional variant may lose at most `0.005` Macro-AUPR and
`0.01` Macro-F1 per paired seed versus B0, while its mean parameter Macro-NMAE
must improve by at least `5%` versus B1. A complete result also requires five
seeds, finite masked-target accounting, all train-only paired-power views, and
terminal/result/abort receipts. Failure stops that branch; no threshold search,
resampling, model enlargement or locked-test rescue is allowed.

## 6. Evidence status

R0, the complete five-seed R1 development run, and the train-only R3 evaluator
are verified with locked-test usage false. R1 B2/B3/B4 are stopped because the
parameter hard gate failed; R2 physical/spectral branches are smoke-only, and
R4 performance ablations were not run because their prerequisites failed.
Existing one-seed/small-sample CSVs remain partial development smoke only.
Final conclusions remain separated into verified, bounded, partial,
stopped/failed, running/uncertain and proposal/not-validated categories.
