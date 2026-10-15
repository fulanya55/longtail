# Six-Class Power Sweep 9-Level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally generate the paired-seed nine-level power-imbalance closed-set datasets for FMB, SawFM, SinFM, MFT, VFT, and ISRJ.

**Architecture:** Reuse the existing manifest-driven closed-set runner and atomic MAT save path. Fix the common nominal JNR at 30 dB, then transform each balanced manifest with a deterministic per-type offset profile before generation. A thin sweep entry point iterates the nine output directories while the saved-file verifier performs power-truth checks without loading multi-gigabyte I/Q arrays.

**Tech Stack:** MATLAB, `matlab.unittest`, MAT v7.3, Git, PowerShell.

---

### Task 1: Level Configuration Contract

**Files:**
- Create: `可配置组合干扰数据生成/create_power_sweep_6class_config.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] Add failing tests that require levels `10:5:50`, reject off-grid levels, fix common JNR at `[30,30]`, preserve the six classes and FMB bandwidth factor `[1.5,3]`, and produce nine unique output directories under `power_sweep_9level_v1`.
- [ ] Run the focused tests and confirm failure because `create_power_sweep_6class_config` does not exist.
- [ ] Implement the level-aware configuration with a saved `cfg.experiment.power_sweep` profile containing fixed codes, swept codes, fixed JNR, current level, standard levels, paired-seed flag and level index.
- [ ] Re-run the focused tests and confirm pass.

### Task 2: Deterministic Manifest JNR Profile

**Files:**
- Create: `可配置组合干扰数据生成/core/apply_power_sweep_jnr_profile.m`
- Create: `可配置组合干扰数据生成/core/build_closedset_manifest.m`
- Modify: `可配置组合干扰数据生成/core/run_closedset_splits.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] Add failing tests for all six singletons and the six-component combination at 10, 30 and 50 dB, including offsets, weak masks, unchanged imbalance flags and unchanged sample seeds across levels.
- [ ] Run the focused tests and confirm failure because the profile helpers do not exist.
- [ ] Implement `apply_power_sweep_jnr_profile` with strict profile validation and per-row code mapping; implement `build_closedset_manifest` as the shared balanced-manifest wrapper; route `run_closedset_splits` through it.
- [ ] Re-run focused and existing manifest tests and confirm pass.

### Task 3: Power-Truth Readback Verification

**Files:**
- Modify: `可配置组合干扰数据生成/core/verify_saved_closedset_split.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] Add a failing small-file test that saves a power-sweep split, verifies valid effective JNR/target/measured power/parameter truth, then corrupts each small metadata field and requires `JammingDataset:SavedFileVerificationFailed`.
- [ ] Run the focused test and confirm failure because current readback does not inspect power truth.
- [ ] Add conditional power-sweep verification using MAT-file reads of small metadata and parameter matrices only; never load `iq_complex` or `data_legacy`.
- [ ] Re-run the focused tests and confirm pass.

### Task 4: Nine-Level Sweep Runner and Status Manifest

**Files:**
- Create: `可配置组合干扰数据生成/generate_closedset_6class_power_sweep.m`
- Create: `可配置组合干扰数据生成/core/write_power_sweep_manifest.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] Add failing tests for default arguments, subset levels, invalid levels, unique 18 output paths, `resume/error/overwrite` forwarding, paired train/test seed signatures and dry-run non-creation.
- [ ] Run focused tests and confirm failure because the public runner does not exist.
- [ ] Implement sequential level execution, compact result structs, paired-seed assertions and root `power_sweep_manifest.csv` writing. Formal results must not retain full manifests or waveform bundles.
- [ ] Re-run focused tests and confirm pass.

### Task 5: Documentation, Regression and Small End-to-End Gate

**Files:**
- Create: `可配置组合干扰数据生成/README_功率不均衡九挡扫描.md`
- Create at generation root: `功率不均衡九挡配对数据集说明.md`

- [ ] Document parameters, directory layout, paired-seed interpretation, MAT fields, JNR truth semantics, FMB Nyquist clipping, resume policy, expected counts and MATLAB commands.
- [ ] Run all MATLAB tests and require all results passed.
- [ ] Run MATLAB Code Analyzer for all `.m` files and require zero issues.
- [ ] Generate a reduced temporary two-level fixture and require atomic save/readback, correct JNR truth and no partial files.

### Task 6: Integration and Formal Generation

**Files:**
- Integrate the verified branch into `G:\论文相关\干扰数据\干扰信号生成程序`.
- Generate under `G:\论文相关\干扰数据\复合干扰数据0716\power_sweep_9level_v1`.

- [ ] Perform a nine-level dry run and verify 63 patterns × 400 samples, identical train seeds across levels, identical test seeds across levels, disjoint train/test seeds, unique output directories and no MAT writes.
- [ ] Run formal generation with `resume`, serially processing level 10 through 50 dB, train then test.
- [ ] After every final file, verify dimensions, manifest/config/schema equality and power truth; ensure no `.partial.mat` remains for completed splits.
- [ ] Produce the root explanation, CSV status inventory, file sizes and SHA-256 hashes.
- [ ] Run an independent final MATLAB readback over all 18 files and report the evidence.
