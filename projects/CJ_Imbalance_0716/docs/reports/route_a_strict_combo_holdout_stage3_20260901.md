# Route-A Stage 3: strict unseen-combination validation (2026-09-01)

Status: completed as development-v2 only. `locked_test_used=false`; E5 was not modified. The run used the RTX 5080 and the same F2_DFMS_Full / real-IQ contract as the label-IF2 screen.

## Protocol

- Training source: 30 dB fit split, identity-isolated from validation.
- Training combinations: all fit samples excluding 15 held-out combination keys.
- Validation: 391 samples from the 15 held-out keys; train/validation combination intersection is empty.
- Training manifest: exactly 1000 samples per seed, effective label IF=2.0.
- Seeds: 2039, 2040, 2041, 2042, 2043.
- Losses: ordinary BCE and fixed Positive-Focal (`gamma=2`).
- No threshold tuning or test-file access.

## Five-seed results

| method | Macro-AUPR | Macro-F1 | exact combination | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.971120 | 0.893862 | 0.628133 | 0.076044 |
| Positive-Focal | 0.975931 | 0.922945 | 0.718159 | 0.055754 |

Positive-Focal improved exact-combination accuracy by 9.00 percentage points and reduced Hamming loss by 0.0203. MFT recall increased by about 0.137 and ISRJ recall by about 0.031.

However, FMB false-positive rate increased by about 0.0285 to approximately 0.103, while SawFM FPR also increased by about 0.0076. Therefore Positive-Focal does not pass the current head-class FPR safety gate and is not promoted as the final method.

## Interpretation

The normal validation gain was not only an artifact of seen-combination reconstruction: Positive-Focal also helps on unseen combinations. The remaining problem is a tradeoff between tail/combination recall and head-class false positives. The next candidate should constrain this tradeoff instead of simply increasing positive response.

## Artifacts

- [Strict-combination manifest summary](../../outputs/route_a_strict_combo_holdout_labelIF2_20260901/manifest_summary.json)
- [BCE/Positive-Focal development summary](../../outputs/route_a_strict_combo_holdout_methods_20260901/summary.json)
- [Additional confirmation summary](../../outputs/route_a_strict_combo_holdout_confirm_20260901/summary.json)
- [Manifest builder](../../experiments/build_route_a_strict_combo_holdout_labelIF2.py)

