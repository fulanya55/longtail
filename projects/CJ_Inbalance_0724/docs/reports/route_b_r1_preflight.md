# Route B R1 baseline preflight

- Status: **verified_smoke_only**
- This is a forward/backward smoke only; no performance result is claimed.
- Development split: `22680` train / `2520` validation, split seed `20260826`.
- Test-v2 created/opened: `False`
- Locked test used: `False`

| Variant | Architecture path | Smoke status |
|---|---|---|
| B0 | classification_only | verified_smoke |
| B1 | shared_backbone_shared_head | verified_smoke |
| B2 | shared_backbone_class_conditioned_head | verified_smoke |
| B3 | class_conditioned_parameter_stop_gradient | verified_smoke |
| B4 | class_conditioned_parameter_loss_warmup | verified_smoke |

The next R1 step is multi-seed development training with the frozen protocol in `configs/route_b_r1_preregister.json`. The current CSV must not be cited as accuracy, AUPR, NMAE, or confirmation evidence.
