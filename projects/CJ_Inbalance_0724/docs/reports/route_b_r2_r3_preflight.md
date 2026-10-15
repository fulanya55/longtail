# Route B R2/R3 preflight

- Status: **verified_audit_and_smoke**
- Test MAT opened: `false`
- Locked test used: `false`

## Auxiliary branches

`{"physical_film": {"feature_shape": [2, 26], "logit_shape": [2, 6], "parameter_shape": [2, 6, 2], "finite": true, "parameter_input_has_no_label_or_target": true}, "spectral_film": {"feature_shape": [2, 19], "logit_shape": [2, 6], "parameter_shape": [2, 6, 2], "finite": true, "parameter_input_has_no_label_or_target": true}}`

## Train-only paired power views

| JNR (dB) | Exists | Labels aligned | receiver_noise_seed aligned |
|---:|:---:|:---:|:---:|
| 10 | True | True | True |
| 15 | True | True | True |
| 20 | True | True | True |
| 25 | True | True | True |
| 30 | True | True | True |
| 35 | True | True | True |
| 40 | True | True | True |
| 45 | True | True | True |
| 50 | True | True | True |

This is an audit/forward smoke only; no accuracy, NMAE, visibility, or power-robustness claim is made.
