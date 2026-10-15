"""Render the fixed H3 evidence boundary from immutable JSON artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{float(value):+.6f}"


def run(args: argparse.Namespace) -> Path:
    observable, summary, audit = _load(args.observable), _load(args.summary), _load(args.audit)
    for name, item in (("observable", observable), ("summary", summary), ("artifact audit", audit)):
        if any(item.get(key, True) for key in ("external_used", "locked_test_used", "test_v2_used", "test_v3_used")):
            raise ValueError(f"H3 {name} violates the test boundary")
    if observable.get("status") != "completed_no_fit_observable" or summary.get("status") != "completed_development_summary" or audit.get("status") != "completed_artifact_audit":
        raise ValueError("H3 report requires completed immutable artifacts")
    if audit.get("record_count") != 15 or not audit.get("t1_t2_equal_parameter_count"):
        raise ValueError("H3 report requires all capacity-matched formal arms")
    comp, checks = summary["comparison"], summary["gate"]["checks"]
    weak_auc = comp["T1_minus_T2_weak_vft_auroc"]
    weak_aupr = comp["T1_minus_T2_weak_vft_aupr"]
    strong = comp["T1_minus_T0_strong_vft_auroc"]
    decision = "通过 H3 的预注册模型门槛" if summary["gate"]["passed"] else "未通过 H3 的预注册模型门槛"
    next_action = (
        "H3 仅提供新的多脉冲、交错 PRI 反事实协议下的前置机制证据；它仍不证明原 0716 单脉冲 test-v3、完整 VTSE-VBF 或六标签统一模型。后续如进入统一模型，必须另行预注册新身份 development 方案。"
        if summary["gate"]["passed"] else
        "按停止规则，不改变时间戳排列、ROI、专家宽度、JNR、epoch、学习率或损失重跑 H3；该物理可观测量存在，但当前目标专属可训练投影未证明其优于等容量置换控制。"
    )
    lines = [
        "# Route A H3：VFT 三层可见性与交错 PRI 可辨识性结果",
        "",
        "日期：2026-08-03  ",
        "证据角色：新的 identity-isolated development-only 机制验证；非 external、非历史 locked test、非 test-v2、非 test-v3。",
        "",
        "## 结论",
        "",
        f"H3 模型比较**{decision}**。{next_action}",
        "",
        "## 物理零拟合前置证据",
        "",
        f"- weak-vs-absent：真实时间戳 AUROC `{observable['weak_vs_absent']['true_auroc']:.6f}`，置换 AUROC `{observable['weak_vs_absent']['permuted_auroc']:.6f}`，差值 `{_fmt(observable['weak_vs_absent']['true_minus_permuted_auroc'])}`，配对 bootstrap 95% CI `{observable['bootstrap']['weak_true_minus_permuted_auroc_95_ci']}`。",
        f"- strong-vs-absent：真实时间戳 AUROC `{observable['strong_vs_absent']['true_auroc']:.6f}`，置换 AUROC `{observable['strong_vs_absent']['permuted_auroc']:.6f}`。",
        f"- 零拟合物理门槛：`{observable['gate']['passed']}`；训练参数数为 `{observable['trained_parameters']}`。",
        "",
        "## 五种子模型对照",
        "",
        "| 对比 | 均值差 | 95% bootstrap CI | 正向种子 |",
        "|---|---:|---:|---:|",
        f"| T1 真时间戳 − T2 置换：weak AUROC | `{_fmt(weak_auc['mean'])}` | `{weak_auc['bootstrap_95_ci']}` | `{weak_auc['positive_seeds']}/5` |",
        f"| T1 真时间戳 − T2 置换：weak AUPR | `{_fmt(weak_aupr['mean'])}` | `{weak_aupr['bootstrap_95_ci']}` | `{weak_aupr['positive_seeds']}/5` |",
        f"| T1 真时间戳 − T0 基线：strong AUROC | `{_fmt(strong['mean'])}` | `{strong['bootstrap_95_ci']}` | `{strong['positive_seeds']}/5` |",
        "",
        "## 预注册门槛",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## 可审计工件与边界",
        "",
        f"- 15 份训练工件、checkpoint 与不可变 development predictions 已由 `{args.audit.as_posix()}` 哈希审计；T1/T2 参数量严格相等。",
        f"- 汇总：`{args.summary.as_posix()}`；零拟合物理结果：`{args.observable.as_posix()}`。",
        "- 校准：无；阈值选择：无；所有结论为 ranking-only development 证据。",
        "- `external_used=false`、`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return args.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observable", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_observable/summary.json"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_development_summary.json"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/route_a_h3_vft_three_strata_results_20260803.md"))
    path = run(parser.parse_args())
    print(json.dumps({"status": "completed_report", "output": str(path)}, ensure_ascii=False))
