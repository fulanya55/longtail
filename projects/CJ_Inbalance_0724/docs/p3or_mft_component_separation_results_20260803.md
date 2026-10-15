# P3o-R: MFT component separability before count extraction

## Decision

P3o-R failed its frozen fresh-identity development gates and is frozen with `stop_no_confirmation_data`. It does not support an MFT component separator as a route to count extraction, C2 attachment, or any predicted-label-gated system. Reserved confirmation identities `1900M/1910M` were not generated or used; no test or locked test was generated, opened, or used.

## Why this was a distinct test

P3c had already established a strong physical upper bound: the exact normalized MFT component, followed by the correct frozen sparse inverse filter, recovered count with MAE `0` and Pearson `1`. P3o-M/N/O/P/Q instead tried to read count features directly from compound I/Q. P3o-R tested the missing link only: can a supervised separator preserve the P3c-relevant component from compound I/Q?

The separator was a 12-width unconditioned complex STFT-mask U-Net. Its only loss was `1 - complex-IQ cosine + 0.2 L1 + 0.25 multi-resolution spectral loss` against an exact, RMS-normalized MFT-only teacher reconstructed from finite delay/amplitude columns. It had no count, ordinal, classification, routing, contrast, threshold, or parameter loss. Count inference remained the zero-trainable-parameter P3c inverse filter.

## Identity and execution boundary

- Train / development receiver-noise seeds: `1880000000` / `1890000000`.
- Each role contains `2,079` paired identities across five regimes and `25/35/50 dB` views.
- The audit found train-development overlap `0`, maximum overlap `0` against `423` prior identity files, and valid power/label contracts.
- One first-batch code-path failure was isolated before any checkpoint or development prediction. The repaired sample-index access was separately frozen; the formal result is the `contractfix` output only.

## Three-seed development result

| seed | waveform cosine | raw MAE | separated MAE | MAE change vs raw | raw Pearson | separated Pearson | separated count support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2341 | 0.9444 | 3.2353 | 3.2786 | -1.34% | 0.0821 | 0.0000 | all 15,840 views predicted 8 |
| 2342 | 0.9448 | 3.2353 | 3.2786 | -1.34% | 0.0821 | 0.0000 | all 15,840 views predicted 8 |
| 2343 | 0.9431 | 3.2353 | 3.2786 | -1.34% | 0.0821 | 0.0000 | all 15,840 views predicted 8 |

All five regimes failed both overall and compound `>=10%` MAE reduction and `Pearson >=0.30` gates in every seed. Ultra-close MAE also worsened (`2.450` raw to `2.475` separated). In contrast, the exact component oracle remained MAE `0`, Pearson `1` in every regime.

## Failure mechanism

The raw compound-IQ to oracle-component cosine was `0.6119`; the separator raised it to about `0.944`. Thus the failure is not a trivial no-learning result. However, the waveform objective rewards global pulse direction and energy agreement, while MFT count is determined by local, discrete inverse-delay support. The learned output preserved the global waveform sufficiently to raise cosine but not the support contrast required by the frozen estimator; its support count saturated to the maximum eight targets.

Therefore, high global reconstruction similarity is not sufficient evidence of count-relevant component preservation. A future component-separation hypothesis must introduce a genuinely new, preregistered support-identifiability contract and fresh identities; it may not retune P3o-R's waveform loss or count extractor on `1880M/1890M`.

## Artifacts

- `configs/p3or_mft_component_separation_preregister_20260803.json`
- `configs/p3or_mft_component_separation_method_freeze_contractfix_20260803.json`
- `configs/p3or_mft_component_separation_codepath_failure_20260803.json`
- `configs/p3or_mft_component_separation_outcome_freeze_20260803.json`
- `outputs/p3or_mft_component_separation_preflight_20260803/data_audit.json`
- `outputs/p3or_mft_component_separation_development_20260803_contractfix/summary.json`
