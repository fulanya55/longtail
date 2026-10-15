"""Fresh-split strict unseen-combination manifests for label-IF2."""
from __future__ import annotations
import hashlib, json
import argparse
from collections import Counter
from pathlib import Path
import sys, h5py, numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cjlt.data.longtail_manifest import build_label_longtail
from cjlt.data.split_builder import combo_key
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, pure_six_indices

def data_root():
    parts = [
        "".join(chr(c) for c in [0x8bba,0x6587,0x76f8,0x5173]),
        "".join(chr(c) for c in [0x5e72,0x6270,0x6570,0x636e]),
        "".join(chr(c) for c in [0x590d,0x5408,0x5e72,0x6270,0x6570,0x636e]) + "0716",
        "".join(chr(c) for c in [0x4e5d,0x7c7b]),
    ]
    return Path("E:/").joinpath(*parts)

def sha(a):
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-seed', type=int, default=260901)
    ap.add_argument('--output-root', default='outputs/route_a_newsplit_strict_combo_labelIF2_20260901')
    args = ap.parse_args()
    out = ROOT / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    source = next(x for x in data_root().rglob("closedset_9class_train.mat") if "level_05" in str(x))
    pure = pure_six_indices(source)
    perm = np.random.default_rng(args.split_seed).permutation(pure)
    fit = np.sort(perm[:4096]); val = np.sort(perm[4096:6144])
    with h5py.File(str(source), "r") as h:
        yp = np.asarray(h["known_label"][:, pure], dtype=np.uint8).T
    labels = yp[:, list(SELECTED_INDICES)]
    yf = labels[np.searchsorted(pure, fit)]; yv = labels[np.searchsorted(pure, val)]
    vc = Counter(combo_key(row) for row in yv)
    held = [k for k,_ in sorted(vc.items(), key=lambda x:(x[1],x[0]))[:15]]
    train_local = np.asarray([i for i,row in enumerate(yf) if combo_key(row) not in held], dtype=np.int64)
    val_local = np.asarray([i for i,row in enumerate(yv) if combo_key(row) in held], dtype=np.int64)
    val_hold = val[val_local]
    role = {
        "schema":"route-a-newsplit-strict-combo-role-v1", "selected_classes":list(CLASS_NAMES),
        "source":str(source.resolve()), "split_seed":args.split_seed,
        "train_candidate_count":int(len(train_local)), "validation_count":int(len(val_hold)),
        "heldout_combinations":held, "train_validation_combo_intersection":[],
        "train_candidate_indices_sha256":sha(fit[train_local]), "validation_indices_sha256":sha(val_hold),
        "locked_test_used":False, "test_files_opened":False, "frozen_e5_modified":False,
    }
    (out/"role_manifest.json").write_text(json.dumps(role, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []; targets = [500,435,379,330,287,250]
    for seed in (2039,2040,2041,2042,2043):
        m = build_label_longtail(yf, train_local, targets, seed, tolerance=.05,
                                 source_split_sha256=sha(fit[train_local]),
                                 target_sample_count=1000, sample_count_tolerance=0.0)
        sel = fit[m.selected_indices]; ys = labels[np.searchsorted(pure, sel)]
        counts = ys.sum(axis=0).astype(int); cc = Counter(combo_key(row) for row in ys)
        payload = {
            "schema":"route-a-newsplit-strict-combo-manifest-v1", "condition":"new_label_IF2_strict_combo",
            "seed":int(seed), "selected_indices":sel.astype(int).tolist(), "selected_count":int(len(sel)),
            "requested_if":2.0, "achieved_label_counts":counts.tolist(), "effective_if":float(counts.max()/counts.min()),
            "combo_counts":{str(k):int(v) for k,v in sorted(cc.items())},
            "validation_indices":val_hold.astype(int).tolist(), "validation_count":int(len(val_hold)),
            "validation_indices_sha256":sha(val_hold), "heldout_combinations":held,
            "locked_test_used":False,
        }
        (out/f"new_label_IF2_strict_combo__seed_{seed}.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        records.append({"seed":int(seed),"selected_count":int(len(sel)),"effective_if":float(counts.max()/counts.min())})
    summary = {"schema":"route-a-newsplit-strict-combo-summary-v1","status":"completed","locked_test_used":False,"frozen_e5_modified":False,"train_candidate_count":int(len(train_local)),"validation_count":int(len(val_hold)),"heldout_combinations":held,"records":records}
    (out/"manifest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"heldout":held,"train":len(train_local),"val":len(val_hold)},ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
