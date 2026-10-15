# Three New Jamming Types and Versioned Registry Design

**Date:** 2026-07-25

**Revised:** 2026-07-26

**Branch:** `codex/four-five-class-datasets`

**Status:** User-approved design, implementation plan written

## 1. Objective

Extend the configurable compound-jamming generator from ten registered atomic
jamming types to thirteen by appending:

| Stable ID | Code | Chinese name |
|---:|---|---|
| 11 | `CombSpectrum` | 梳状谱干扰 |
| 12 | `ClutterPulse` | 杂乱脉冲干扰 |
| 13 | `SpectralDispersion` | 频谱弥散干扰 |

The three types shall be physically coupled to the existing radar reference
LFM signal, expose complete parameter-extraction truth, support balanced,
imbalanced and open-set generation, and participate in the existing component
power/JNR pipeline.

The extension must preserve the exact IDs, parameter ordering, file names and
saved configuration structure of existing four-class, five-class and six-class
datasets so that their current MAT files continue to pass `resume` readback.

## 2. Scope and Non-goals

### 2.1 In scope

- A versioned ten-class/v1 and thirteen-class/v2 registry.
- Three new atomic MATLAB generators.
- New v2 default parameters and physical validation.
- Fixed parameter-schema slots for all variable-cardinality truth.
- Thirteen-column `label_all` for v2 data.
- Exact ten-column `label_all` compatibility for v1 data.
- Tests covering waveform physics, truth encoding, power calibration,
  reproducibility, version compatibility and saved-data readback.
- Documentation explaining how users select modes and modify parameters.

### 2.2 Out of scope

- Regenerating or migrating existing MAT files.
- Changing the first ten type IDs, names, generators or parameter schemas.
- Repairing the previously identified semantic overlap between MFT and
  `RangeFakeTarget`.
- Redesigning the existing `Traction` model.
- Generating a formal thirteen-class dataset in this implementation task.
- Adding phase-modulated ISRJ modes.

## 3. Registry and Saved-data Compatibility

### 3.1 Version rules

The program shall support two deterministic registry versions:

- **v1:** the existing ten entries, unchanged byte-for-byte in semantic content;
- **v2:** the same ten entries in the same order, followed by IDs 11--13.

`build_jamming_registry(cfg)` selects the registry as follows:

1. If `cfg.registry.version` is present, it must be integer `1` or `2` and the
   matching registry is returned.
2. If the configuration has no `registry.version`, it is treated as legacy v1.
3. Calling `build_jamming_registry()` without a configuration returns v2.

The first ten v2 entries must be constructed by the same shared v1 builder,
not copied into a second independent definition. This prevents registry drift.

### 3.2 Default configuration API

`create_default_config` gains one optional version argument:

```matlab
cfg = create_default_config();   % v2, thirteen classes
cfg = create_default_config(2);  % v2, thirteen classes
cfg = create_default_config(1);  % exact legacy v1 template
```

The v1 return value must preserve the old configuration structure:

- no `cfg.registry` field;
- only the original ten parameter structures;
- original ten-element default component-weight vector;
- original 10 MHz radar sampling rate and all other old defaults.

This exact structural preservation is required because saved-file readback uses
configuration equality, not only equivalent behavior.

The v2 return value explicitly contains:

```matlab
cfg.registry.version = 2;
```

and includes all thirteen type parameters.

### 3.3 Legacy experiment configurations

`create_closedset_6class_config` must use `create_default_config(1)`. The
four-class and five-class configurations inherit from this six-class template,
and the corresponding power-sweep configurations inherit from those subset
configurations. Consequently:

- old four/five/six-class MAT files retain ten-column `label_all`;
- their parameter schema and saved `config` remain exactly compatible;
- `resume` continues to validate and reuse them;
- no new type parameter structures leak into legacy saved configurations.

### 3.4 v2 default known/unknown partition

The v2 default enables all thirteen classes. Its known set contains the five
previous default known classes plus all three new classes:

```text
FMB, AMA, SawFM, MFT, ISRJ,
CombSpectrum, ClutterPulse, SpectralDispersion
```

The v2 default unknown set remains the five original types:

```text
SinFM, SN, VFT, RangeFakeTarget, Traction
```

Thus v2 defaults to eight known and five unknown classes. Users may continue to
edit `enabled`, `known` and `unknown` before normalization.

## 4. Radar Sampling Strategy

Both the v1 legacy configuration and the v2 default retain:

```matlab
cfg.radar.fs_Hz = 10e6;
```

The 100 microsecond pulse width, 300 microsecond PRI, 5 MHz LFM bandwidth,
10 GHz carrier and four-pulse default record also remain unchanged. Therefore,
the samples per record and per-record storage size do not increase because of
the sampling strategy. Existing configured sample counts are not increased or
multiplied automatically.

Before any of the three new generators processes the intercepted analytic LFM,
the reference is mixed by the negative midpoint of its discrete frequency
support so that the approximately 5 MHz support is centered at zero. This is a
complex-baseband frequency translation; it preserves chirp slope and relative
phase structure while providing symmetric Nyquist headroom.

With a centered reference support of approximately `[-2.5, 2.5] MHz`, the
approved default domains remain representable at 10 MHz:

- CombSpectrum's largest approved tooth offset is approximately `+/-2.4 MHz`,
  producing an outer support of approximately `+/-4.9 MHz`;
- ClutterPulse's frequency offset is limited to `+/-1 MHz`, producing an outer
  support of approximately `+/-3.5 MHz`;
- every SMSP subpulse sweeps the original approximately 5 MHz reference band.
  Increasing the integer subpulse count from 2 to 8 increases chirp rate, not
  occupied bandwidth, so its support remains approximately `+/-2.5 MHz`.

Normalization and generation both apply a strict Nyquist guard to the realized
discrete support. A request at or beyond the Nyquist boundary, or one made
infeasible by user-edited radar or jammer parameters, raises
`JammingGenerator:InvalidPhysicalParameter`; no generator silently aliases or
clips the requested spectrum. SMSP therefore does not use an independent
continuous compression-ratio parameter.

## 5. `CombSpectrum` Model

### 5.1 Physical construction

Let the reference LFM train be `s(t)`. Its discrete LFM frequency support is
first recentered to zero by removing the midpoint of the reference support,
forming `s0(t)`. For `K` teeth, the clean jamming waveform is

\[
j(t)=\sum_{k=1}^{K} a_k s_0(t)
\exp\left(j\left(2\pi f_k t+\phi_k\right)\right),
\]

with

\[
f_k=f_c+\left(k-\frac{K+1}{2}\right)\Delta f.
\]

This yields frequency-shifted LFM replicas whose offsets form an exact comb.
Dechirping with `conj(s0(t))` exposes discrete tones at the saved `f_k`, which
provides a robust physics test even when the shifted LFM spectra overlap.

### 5.2 Default configuration

```matlab
cfg.params.CombSpectrum.num_teeth = [3, 8];
cfg.params.CombSpectrum.spacing_Hz = [0.2e6, 0.6e6];
cfg.params.CombSpectrum.center_offset_Hz = [-0.3e6, 0.3e6];
cfg.params.CombSpectrum.amplitude = [0.5, 1.5];
cfg.params.CombSpectrum.amplitude_mode = 'equal';
cfg.params.CombSpectrum.max_teeth = 8;
```

`amplitude_mode` accepts:

- `equal`: every active tooth has amplitude one;
- `random`: each active tooth independently samples the configured amplitude
  range.

Each tooth receives an independent uniform phase in `[0, 2*pi)`.

### 5.3 Nyquist handling

For every sampled tuple, the full shifted reference support must stay strictly
inside the Nyquist guard. The generator samples only from the feasible
intersection of the configured center/spacing ranges and the reference-aware
support. If no tuple with at least three teeth is feasible, it raises
`JammingGenerator:InvalidPhysicalParameter`. It must not silently alias or
pretend that an infeasible request was realized.

### 5.4 Arrival delay

The common arrival delay is quantized to integer samples and applied with a
zero-boundary delay to the full clean record. No tail samples wrap to the front.

### 5.5 Truth schema

The fixed fields are:

```text
JNR_dB
num_teeth
spacing_Hz
center_offset_Hz
amplitude_mode_code
arrival_delay_s
frequency_01_Hz ... frequency_08_Hz
amplitude_01 ... amplitude_08
phase_01_rad ... phase_08_rad
```

Mode codes are `0=equal`, `1=random`. Tooth frequencies are baseband offsets
after reference recentering. Unused slots are `NaN` and have a false parameter
mask.

## 6. `ClutterPulse` Model

### 6.1 Physical construction

Within each PRI, the jammer creates a random group of gated reference-LFM
pulses. The gated reference is the same zero-centered complex-baseband LFM
defined in Section 4, so the approved `+/-1 MHz` pulse offsets remain inside the
10 MHz Nyquist guard. For pulse `m` in PRI `p`,

\[
j_{p,m}(t)=a_{p,m}\,w_{p,m}(t)\,
s_0(t-\tau_{p,m})\,
\exp\left(j\left(2\pi f_{p,m}t+\phi_{p,m}\right)\right),
\]

where `w` is a rectangular gate with the sampled width. Pulses may overlap and
are summed coherently as complex samples. They are based on the existing LFM,
not synthesized as unrelated standalone rectangles.

### 6.2 Modes

`cfg.params.ClutterPulse.mode` accepts:

- `independent`: every PRI independently samples count, start, width,
  amplitude, frequency offset and phase;
- `repeated`: the first PRI samples one layout and all later PRIs repeat the
  same intra-PRI layout and all pulse parameters.

Mode codes are `0=independent`, `1=repeated`. The default is `independent`.

### 6.3 Default configuration and capacity

```matlab
cfg.params.ClutterPulse.pulses_per_pri = [3, 8];
cfg.params.ClutterPulse.pulse_width_s = [2e-6, 20e-6];
cfg.params.ClutterPulse.relative_amplitude = [0.5, 1.5];
cfg.params.ClutterPulse.frequency_offset_Hz = [-1e6, 1e6];
cfg.params.ClutterPulse.mode = 'independent';
cfg.params.ClutterPulse.max_pulses_per_pri = 8;
cfg.params.ClutterPulse.max_total_pulses = 32;
```

Normalization requires

```text
radar.num_pulses * pulses_per_pri(2) <= max_total_pulses
```

so the default four-PRI record exactly fits 32 slots. Users changing pulse
count or radar pulse count must also respect this fixed truth capacity.

Each start time is sampled so the pulse gate remains within its PRI. Overlap is
allowed. The common arrival delay is then applied with zero boundaries to the
complete clean record; validation requires that the delayed support still fits
the record.

### 6.4 Truth schema

Summary fields:

```text
JNR_dB
mode_code
total_pulse_count
mean_width_s
duty_cycle
mean_amplitude
minimum_frequency_Hz
maximum_frequency_Hz
arrival_delay_s
```

`duty_cycle` is the union of occupied sample positions divided by total record
length, so overlap is counted once.

The 32 slots are flattened in PRI-major order and then start-time order within
each PRI. `start_XX_s` is absolute time from the beginning of the record:

```text
start_01_s ... start_32_s
width_01_s ... width_32_s
amplitude_01 ... amplitude_32
frequency_01_Hz ... frequency_32_Hz
phase_01_rad ... phase_32_rad
```

Unused slots are `NaN` and have a false parameter mask. In repeated mode, the
per-PRI absolute starts differ by integer PRI offsets while widths, amplitudes,
frequency offsets, phases and intra-PRI start positions repeat exactly.

## 7. `SpectralDispersion` SMSP Model

### 7.1 Physical construction

The approved model is the standard DRFM smeared-spectrum construction. Let the
zero-centered intercepted reference LFM have pulse width `T`, bandwidth `B` and
chirp rate `mu = B/T`. Sample one integer subpulse count `M` from 2 through 8.
The ideal continuous-time SMSP quantities are

```text
subpulse_width = T / M
subpulse_chirp_rate = M * mu
subpulse_bandwidth = B
total_output_duration = M * subpulse_width = T
```

Let `T_J = T/M`, `mu_J = B/T_J = M*mu` and let
`u_m = t - m*T_J` be local time in subpulse `m`. Before the common arrival
delay, the continuous model is

\[
j_m(t)=A_J\,\mathbf{1}_{[0,T_J)}(u_m)
\exp\left(j\pi\mu_J(u_m-T_J/2)^2+j\phi_J\right),
\]

\[
j_{\mathrm{SMSP}}(t)=\sum_{m=0}^{M-1}j_m(t).
\]

Each subpulse is an identical short LFM sweep over the complete original band,
from approximately `-B/2` to `+B/2`. Its local phase resets coherently at each
subpulse boundary; the implementation samples one global initial phase rather
than independent random phases for different subpulses. The `M` identical
subpulses are placed consecutively, producing `M` frequency-reset sweeps within
one original pulse width.

The discrete generator derives `B`, `T`, center frequency and phase convention
from the active reference LFM. It calculates one integer subpulse sample count
as the nearest positive sample count to `fs*T/M`, synthesizes the reference-
coupled short chirp, and repeats those exact complex samples `M` times. Any
sample-quantization difference from the ideal `T/M` is written to truth rather
than hidden.

The complete waveform is placed in every PRI at the common arrival delay. All
remaining record samples are zero and arrival delay is never circular. The
model is a coherent repeater-jamming transformation of the intercepted
reference, not random phase modulation, FMB-like noise modulation, or whole-
pulse time scaling.

### 7.2 Default configuration

```matlab
cfg.params.SpectralDispersion.num_subpulses = [2, 8];
cfg.params.SpectralDispersion.max_subpulses = 8;
```

`num_subpulses` is an integer range. The sampled value `M` simultaneously
defines the repeat count, ideal time-compression factor and ideal chirp-rate
multiplier. There is no separately sampled continuous compression ratio.

The realized discrete values are

```text
subpulse_width_s = subpulse_samples / fs
chirp_rate_multiplier = pulse_width_s / subpulse_width_s
subpulse_chirp_rate_Hz_per_s = bandwidth_Hz / subpulse_width_s
output_duration_s = M * subpulse_width_s
```

Each subpulse retains the reference bandwidth, so increasing `M` does not expand
the occupied band. Validation requires at least two samples per subpulse, the
reference band to remain strictly inside the 10 MHz Nyquist guard, and

```text
arrival_delay_s + output_duration_s <= pri_s.
```

An infeasible request raises `JammingGenerator:InvalidPhysicalParameter`.

### 7.3 Truth schema

```text
JNR_dB
num_subpulses
subpulse_width_s
chirp_rate_multiplier
subpulse_chirp_rate_Hz_per_s
subpulse_bandwidth_Hz
output_duration_s
arrival_delay_s
initial_phase_rad
subpulse_start_01_s ... subpulse_start_08_s
```

`num_subpulses` is the sampled integer `M`. `chirp_rate_multiplier`, subpulse
width, chirp rate and output duration are realized discrete truth after sample
quantization. Subpulse starts refer to the consecutive SMSP waveform before the
common arrival delay. Unused slots are `NaN` with false masks.

## 8. Common Generation Pipeline

All three types follow the existing pipeline:

```text
configuration normalization
-> v1/v2 registry selection
-> reference LFM construction
-> clean atomic waveform and truth generation
-> component JNR/power calibration
-> compound summation
-> one receiver-noise addition
-> label and parameter-target encoding
```

Common rules:

- only the passed scalar `RandStream` may be consumed;
- a fixed seed reproduces waveform and truth exactly;
- outputs are finite `1 x sample_length` complex-double rows;
- new types use common `JNR_dB`, unlike `Traction`;
- component power imbalance is applied by the existing combination layer and
  final effective JNR is written back into parameter truth;
- requested continuous quantities are reported as realized discrete truth
  after sample quantization;
- unsupported modes and infeasible physical domains use stable errors;
- no new generator adds receiver background noise internally.

## 9. Parameter-schema Size

The existing v1 parameter schema has 79 columns. The new fixed fields add:

| Type | Added columns |
|---|---:|
| `CombSpectrum` | 30 |
| `ClutterPulse` | 169 |
| `SpectralDispersion` | 17 |
| Total added | 216 |

The v2 parameter schema therefore has 295 columns. v1 remains exactly 79.

## 10. Validation and Error Handling

Configuration normalization shall validate only parameter structures belonging
to the selected registry version and enabled types. It must validate:

- version is 1 or 2;
- integer cardinality ranges and fixed capacities;
- text modes and stable mode codes;
- strictly positive widths, amplitudes and spacings;
- ordered finite frequency ranges;
- reference-aware Nyquist support;
- zero-boundary delayed support within the record;
- clutter total slot capacity;
- SMSP integer subpulse capacity, minimum samples per subpulse, realized output
  duration and reference-band feasibility.

Generator-level defensive checks repeat critical physical constraints so a
caller cannot bypass normalization and silently create invalid waveforms.

## 11. Test Design

### 11.1 Version compatibility

- `create_default_config(1)` reproduces the old ten-class structure and contains
  no v2 fields.
- `build_jamming_registry(create_default_config(1))` returns exactly the old ten
  entries and old 79-column schema.
- `create_default_config()` and `(2)` return thirteen entries and a 295-column
  schema.
- v2 IDs 1--10 equal v1 IDs, codes, Chinese names, generator handles, parameter
  names and units.
- IDs 11--13 are stable and appended.
- legacy four/five/six-class dry runs still produce ten-column `label_all`.
- existing generated four/five/six-class MAT files pass `resume` without
  regeneration or configuration mismatch.
- new v2 balanced, imbalanced and open-set smoke datasets use thirteen-column
  `label_all`.

### 11.2 Common generator contract

For all thirteen generators:

- output length, class, complex shape, finite values and positive power;
- exact fixed-seed reproducibility;
- no global RNG mutation;
- exact registry truth-field set;
- no internal receiver noise;
- component-pipeline target/measured power within configured tolerance.

### 11.3 `CombSpectrum`

- equal and random modes;
- 3 and 8 tooth capacity boundaries;
- tooth frequency, amplitude and phase slots plus `NaN` padding;
- dechirped FFT peak spacing equals saved spacing within one FFT bin;
- saved center equals the mean of active tooth offsets;
- all shifted reference supports stay inside the Nyquist guard;
- impossible center/spacing/count domains fail explicitly;
- zero-boundary arrival delay produces leading zeros and no wraparound.

### 11.4 `ClutterPulse`

- independent and repeated modes;
- per-PRI count range, 32-slot boundary and capacity rejection;
- pulse widths, starts, amplitudes, frequencies and phases match detailed truth;
- slots follow PRI-major/start-time ordering;
- repeated mode reproduces the same intra-PRI layout and parameters;
- independent mode produces at least two differing PRI layouts for a fixed
  regression seed;
- duty cycle equals union occupancy, including overlap cases;
- zero-boundary delay and record-support checks.

### 11.5 `SpectralDispersion`

- 2 and 8 integer subpulse boundaries;
- exactly `M` identical, coherently phase-reset local subpulse waveforms;
- consecutive subpulse starts in increasing time order with fixed capacity and
  `NaN` padding;
- ideal width `T/M` and realized sample-quantized width, chirp-rate multiplier,
  chirp rate and total duration truth;
- every subpulse sweeps the original reference band and increasing `M` does not
  increase occupied bandwidth;
- default 10 MHz sampling supports all integer values from 2 through 8;
- a configuration producing fewer than two samples per subpulse is rejected;
- zero padding outside the SMSP output;
- no circular leakage between record end and beginning.

### 11.6 Distinctness and regression

- the three new types must not generate exactly equal waveforms to one another
  or to representative existing types under matched seeds and fixed parameters;
- all existing tests remain passing after expectations are version-aware;
- full MATLAB Code Analyzer reports zero issues.

## 12. Documentation

User documentation shall explain:

- v1/v2 registry selection;
- old MAT compatibility and why old data remains ten-column;
- new default known/unknown sets;
- the common 10 MHz sampling rate, zero-centered reference convention and SMSP
  integer subpulse-count range of 2--8;
- editable parameters and mode strings;
- fixed slot semantics and `NaN` masks;
- the distinction between requested and realized discrete truth;
- examples generating each new type alone and in combinations.

## 13. Acceptance Criteria

Implementation is accepted only when:

1. v1 remains exactly compatible with existing four/five/six-class MAT files.
2. v2 exposes thirteen stable labels and 295 parameter columns.
3. all three generators satisfy their approved physical models and truth
   schemas.
4. v2 remains at 10 MHz, standard SMSP uses one integer subpulse-count parameter
   from 2 through 8 with no independent continuous compression ratio, and
   existing configured record and dataset sample counts are not increased
   automatically.
5. no generator aliases, wraps an arrival delay, or silently violates a fixed
   capacity.
6. all old and new MATLAB tests pass.
7. Code Analyzer reports zero issues.
8. program documentation is complete and comments explain all nontrivial
   physics, quantization and compatibility decisions.
