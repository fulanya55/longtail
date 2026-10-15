"""Aggregate the current Route-A long-tail evidence into one auditable ledger."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(pattern):
    patterns = pattern if isinstance(pattern, (list, tuple)) else [pattern]
    paths = []
    for item in patterns:
        paths.extend(glob.glob(str(ROOT / item)))
    unique = sorted(set(paths))
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in unique]

def agg(rows):
    keys = ["macro_aupr","macro_f1","combination_exact_accuracy","hamming_loss"]
    out = {"n": len(rows), "seeds": sorted(int(r["seed"]) for r in rows)}
    for k in keys:
        v = np.asarray([r["validation"][k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v)>1 else 0.0, "values": v.tolist()}
    out["elapsed_seconds_mean"] = float(np.mean([r.get("elapsed_seconds", float("nan")) for r in rows]))
    out["parameters"] = int(rows[0].get("total_parameters", 0)) if rows else None
    classes = ["FMB","AMA","SawFM","SinFM","MFT","ISRJ"]
    out["per_label"] = {}
    for c in classes:
        out["per_label"][c] = {}
        for k in ["aupr","recall","negative_fpr"]:
            out["per_label"][c][k] = float(np.mean([r["validation"]["per_label"][c][k] for r in rows]))
    return out

def main():
    outdir = ROOT / "outputs" / "route_a_longtail_goal_aggregate_20260901"; outdir.mkdir(parents=True, exist_ok=True)
    groups = {
        "normal_labelIF2_BCE": ["outputs/route_a_bce_boundary_screen_20260901/label_IF2/label_IF2__seed_*.json", "outputs/route_a_bce_boundary_confirm_20260901/label_IF2/label_IF2__seed_*.json"],
        "normal_labelIF2_CB_BCE": "outputs/route_a_longtail_methods_labelIF2_screen_20260901/label_IF2/CB_BCE/label_IF2__CB_BCE__seed_*.json",
        "normal_labelIF2_ASL": "outputs/route_a_longtail_methods_labelIF2_screen_20260901/label_IF2/ASL/label_IF2__ASL__seed_*.json",
        "normal_labelIF2_PositiveFocal": ["outputs/route_a_longtail_methods_labelIF2_screen_20260901/label_IF2/PositiveFocal/label_IF2__PositiveFocal__seed_*.json", "outputs/route_a_longtail_methods_labelIF2_confirm_20260901/label_IF2/PositiveFocal/label_IF2__PositiveFocal__seed_*.json"],
        "normal_labelIF2_DRW": "outputs/route_a_longtail_methods_labelIF2_drw_20260901/label_IF2/DRW/label_IF2__DRW__seed_*.json",
        "normal_labelIF2_ComboSampler": "outputs/route_a_longtail_methods_labelIF2_combosampler_20260901/label_IF2/ComboSampler/label_IF2__ComboSampler__seed_*.json",
        "strict_BCE": ["outputs/route_a_strict_combo_holdout_methods_20260901/label_IF2_strict_combo/BCE/label_IF2_strict_combo__BCE__seed_*.json", "outputs/route_a_strict_combo_holdout_confirm_20260901/label_IF2_strict_combo/BCE/label_IF2_strict_combo__BCE__seed_*.json"],
        "strict_PositiveFocal": ["outputs/route_a_strict_combo_holdout_methods_20260901/label_IF2_strict_combo/PositiveFocal/label_IF2_strict_combo__PositiveFocal__seed_*.json", "outputs/route_a_strict_combo_holdout_confirm_20260901/label_IF2_strict_combo/PositiveFocal/label_IF2_strict_combo__PositiveFocal__seed_*.json"],
        "strict_CB_BCE": "outputs/route_a_strict_combo_cb_asl_screen_20260901/label_IF2_strict_combo/CB_BCE/label_IF2_strict_combo__CB_BCE__seed_*.json",
        "strict_ASL": "outputs/route_a_strict_combo_cb_asl_screen_20260901/label_IF2_strict_combo/ASL/label_IF2_strict_combo__ASL__seed_*.json",
        "strict_PFmix25": ["outputs/route_a_strict_combo_pf_mix25_screen_20260901/label_IF2_strict_combo/PFmix25/label_IF2_strict_combo__PFmix25__seed_*.json", "outputs/route_a_strict_combo_pf_mix25_confirm_20260901/label_IF2_strict_combo/PFmix25/label_IF2_strict_combo__PFmix25__seed_*.json"],
        "strict_LogitAdj": "outputs/route_a_strict_combo_logitadj_screen_20260901/label_IF2_strict_combo/LogitAdj/label_IF2_strict_combo__LogitAdj__seed_*.json",
        "strict_PF_HN": "outputs/route_a_strict_combo_pf_hn_screen_20260901/label_IF2_strict_combo/PF_HN/label_IF2_strict_combo__PF_HN__seed_*.json",
        "strict_ComboMargin": "outputs/route_a_strict_combo_margin_screen_20260901/label_IF2_strict_combo/ComboMargin/label_IF2_strict_combo__ComboMargin__seed_*.json",
        "strict_TailPositiveFocal": "outputs/route_a_strict_combo_tailfocal_screen_20260901/label_IF2_strict_combo/TailPositiveFocal/label_IF2_strict_combo__TailPositiveFocal__seed_*.json",
        "newsplit1_strict_BCE": ["outputs/route_a_newsplit_strict_combo_methods_screen_20260901/new_label_IF2_strict_combo/BCE/*.json", "outputs/route_a_newsplit_strict_combo_methods_confirm_20260901/new_label_IF2_strict_combo/BCE/*.json"],
        "newsplit1_strict_PositiveFocal": ["outputs/route_a_newsplit_strict_combo_methods_screen_20260901/new_label_IF2_strict_combo/PositiveFocal/*.json", "outputs/route_a_newsplit_strict_combo_methods_confirm_20260901/new_label_IF2_strict_combo/PositiveFocal/*.json"],
        "newsplit2_strict_BCE": ["outputs/route_a_newsplit2_strict_combo_methods_screen_20260901/new_label_IF2_strict_combo/BCE/*.json", "outputs/route_a_newsplit2_strict_combo_methods_confirm_20260901/new_label_IF2_strict_combo/BCE/*.json"],
        "newsplit2_strict_PositiveFocal": ["outputs/route_a_newsplit2_strict_combo_methods_screen_20260901/new_label_IF2_strict_combo/PositiveFocal/*.json", "outputs/route_a_newsplit2_strict_combo_methods_confirm_20260901/new_label_IF2_strict_combo/PositiveFocal/*.json"],
        "newsplit2_strict_PF_HN": "outputs/route_a_newsplit2_strict_combo_pfhn_screen_20260901/new_label_IF2_strict_combo/PF_HN/*.json",
    }
    summary = {"schema":"route-a-longtail-goal-aggregate-v1","status":"completed","locked_test_used":False,"frozen_e5_modified":False,"groups":{}}
    for name, pattern in groups.items():
        rows = load(pattern)
        summary["groups"][name] = agg(rows) if rows else {"n":0,"missing":pattern}
    # Explicit decisions are protocol-level, not inferred from a single metric.
    summary["decisions"] = {
        "BCE_boundary": "label_IF2 confirmed; strict unseen-combination validation required for claims",
        "promoted_method": None,
        "PositiveFocal": "cross-split development candidate: improves unseen-combination metrics on two independent splits, but split-2 ISRJ FPR increase violates the preregistered safety gate",
        "PF_HN": "stopped after split-2 screen: negligible combination gain with worse Macro-F1/Hamming",
        "mixed_longtail": "exact independent IF=2 and CF=4/8 infeasible under current source/budget",
        "locked_test": "not opened",
    }
    (outdir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((outdir/"summary.json").resolve()),"locked_test_used":False,"frozen_e5_modified":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((outdir/"summary.json").resolve()),"groups":{k:v.get("n",0) for k,v in summary["groups"].items()}},ensure_ascii=False),flush=True)

if __name__ == "__main__": main()
