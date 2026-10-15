# VFT--ISRJ source-v1 high-resolution STFT projection result

## Result

The source task contains substantial time-phase organisation: the projected
32x32 descriptor beats its equal-dimensional phase-scrambled control on early
AUROC by `+0.14871` (95% CI `[+0.11274, +0.18001]`) and on AUPR by `+0.15539`.
However, greater native STFT resolution does **not** improve separation beyond
the existing 16x16 descriptor after classifier capacity is matched.

| Early comparison | Delta AUROC, 95% CI | Delta AUPR, 95% CI |
|---|---:|---:|
| High32Projected minus Native16 | -0.00049, [-0.00988, +0.01159] | -0.03261, [-0.04532, -0.01927] |
| High32Projected minus High32PhaseScrambledProjected | +0.14871, [+0.11274, +0.18001] | +0.15539, [+0.12459, +0.18618] |

Native16 achieves early AUROC/AUPR `0.65934 / 0.52791`; the equal-classifier-
capacity High32Projected arm reaches `0.65885 / 0.49530`. The latter's
validation-to-early AUROC gap is `-0.00179`, so this negative finding is not a
fit-adjacent validation artifact.

## Decision and mechanism implication

The phase-order control gate passes, but the key resolution gate fails: the
high-resolution representation has only one of five early AUROC gains of
`>=0.020`, no mean gain, and materially worse AUPR than Native16. Therefore
we stop the high-resolution/projection route and do not generate a
development-v13 candidate from it.

The result narrows the OOD diagnosis. VFT--ISRJ confusion is not supported as
an aliasing problem that can be fixed simply by preserving more STFT cells at
the same classifier capacity. The 16x16 grid already captures the useful
source-level time-phase information more efficiently than the fixed projected
32x32 grid. Further grid-size searches or high-resolution network expansion
are prohibited by this failed pre-registered mechanism test.

## Reproducibility and boundary

- Plan: [route_a_vft_isrj_source_v1_highres_stft_projection_plan_20260801.md](../plans/route_a_vft_isrj_source_v1_highres_stft_projection_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_isrj_source_v1_highres_stft_projection.py`
- Summary: `outputs/tapc/route_a_vft_isrj_source_v1_highres_stft_projection/route_a_vft_isrj_source_v1_highres_stft_projection_summary.json`
- Source SHA-256: `c6e041cf286a1edc26477b4595963d0802ee990201cee05e617b3001008f2cd5`.
- Projector SHA-256: `8c2d82d98a1e6414faca8193202ef4c3263976e7db08c799f4a531576b228d59`.

All records state `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
