"""Build a strict unseen-combination development split for label-IF=2."""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
import sys
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import build_label_longtail
from cjlt.data.split_builder import combo_key
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, pure_six_indices


def discover_data_root() -> Path:
    parts = [
        "".join(chr(c) for c in [0x8bba,0x6587,0x76f8,0x5173]),
        "".join(chr(c) for c in [0x5e72,0x6270,0x6570,0x636e]),
        "".join(chr(c) for c in [0x590d,0x5408,0x5e72,0x6270,0x6570,0x636e])+"0716",
        "".join(chr(c) for c in [0x4e5d,0x7c7b]),
    ]
    return Path("E:/").joinpath(*parts)


def sha_idx(a): return hashlib.sha256(np.asarray(a,dtype="<i8").tobytes()).hexdigest()


def main():
    out = PROJECT_ROOT / "outputs" / "route_a_strict_combo_holdout_labelIF2_20260901"; out.mkdir(parents=True, exist_ok=True)
    root = discover_data_root(); source = next(x for x in root.rglob("closedset_9class_train.mat") if "level_05" in str(x))
    pure = pure_six_indices(source); perm = np.random.default_rng(260828).permutation(pure); fit = np.sort(perm[:4096]); val = np.sort(perm[4096:6144])
    with h5py.File(str(source), "r") as h: labels_pure = np.asarray(h["known_label"][:, pure], dtype=np.uint8).T
    labels = labels_pure[:, list(SELECTED_INDICES)]; fit_local = np.searchsorted(pure, fit); val_local = np.searchsorted(pure, val); yfit = labels[fit_local]; yval = labels[val_local]
    val_keys = [combo_key(row) for row in yval]; vc = Counter(val_keys)
    held = [k for k,_ in sorted(vc.items(), key=lambda x:(x[1],x[0]))[:15]]
    train_local = np.asarray([i for i,row in enumerate(yfit) if combo_key(row) not in held], dtype=np.int64)
    val_hold_local = np.asarray([i for i,key in enumerate(val_keys) if key in held], dtype=np.int64)
    val_hold = val[val_hold_local]
    role = {
        "schema":"route-a-strict-combo-holdout-role-v1","selected_classes":list(CLASS_NAMES),
        "source":str(source.resolve()),"fit_split_seed":260828,"train_candidate_count":int(len(train_local)),
        "validation_count":int(len(val_hold)),"heldout_combinations":held,
        "train_combinations":sorted(set(combo_key(row) for row in yfit[train_local])),
        "combo_intersection":sorted(set(combo_key(row) for row in yfit[train_local]) & set(combo_key(row) for row in yval[val_hold_local])),
        "train_candidate_indices_sha256":sha_idx(fit[train_local]),"validation_indices_sha256":sha_idx(val_hold),
        "locked_test_used":False,"test_files_opened":False,"frozen_e5_modified":False,
    }
    (out/"role_manifest.json").write_text(json.dumps(role,ensure_ascii=False,indent=2),encoding="utf-8")
    targets=[500,435,379,330,287,250]
    records=[]
    for seed in (2039,2040,2041,2042,2043):
        m=build_label_longtail(yfit,train_local,targets,int(seed),tolerance=.05,source_split_sha256=sha_idx(fit[train_local]),target_sample_count=1000,sample_count_tolerance=0.0)
        selected=fit[m.selected_indices]; ysel=labels[np.searchsorted(pure,selected)]; counts=ysel.sum(0).astype(int)
        payload={"schema":"route-a-strict-combo-holdout-manifest-v1","condition":"label_IF2_strict_combo","seed":int(seed),"axis":"label_only","requested_if":2.0,"target_label_counts":targets,"achieved_label_counts":counts.tolist(),"effective_if":float(counts.max()/counts.min()),"selected_indices":selected.astype(int).tolist(),"selected_count":int(len(selected)),"selected_indices_sha256":sha_idx(selected),"combo_counts":{str(k):int(v) for k,v in sorted(Counter(combo_key(row) for row in ysel).items())},"validation_indices":val_hold.astype(int).tolist(),"validation_count":int(len(val_hold)),"validation_indices_sha256":sha_idx(val_hold),"heldout_combinations":held,"combo_intersection":[],"locked_test_used":False}
        path=out/f"label_IF2_strict_combo__seed_{seed}.json"; path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); records.append(payload)
    summary={"schema":"route-a-strict-combo-holdout-summary-v1","status":"completed","locked_test_used":False,"frozen_e5_modified":False,"train_candidate_count":int(len(train_local)),"validation_count":int(len(val_hold)),"heldout_combinations":held,"combo_intersection":[],"records":[{"seed":r["seed"],"selected_count":r["selected_count"],"effective_if":r["effective_if"],"validation_count":r["validation_count"]} for r in records]}
    (out/"manifest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"heldout":held,"train":len(train_local),"val":len(val_hold)},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
