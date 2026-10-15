"""Build fixed-budget IF sweep manifests from the existing six-class C1 fit."""
from __future__ import annotations
import hashlib, json
from collections import defaultdict
from pathlib import Path
import sys
import h5py, numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cjlt.data.split_builder import combo_key

ROLE = ROOT / "outputs/route_a_c1_baseline_full_20260826/role_manifest.json"
OUT = ROOT / "outputs/route_a_if_sweep_20260902/manifests"
NAMES = ["FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ"]
SEEDS = (2039, 2040, 2041, 2042, 2043)


def sha(values): return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    role = json.loads(ROLE.read_text(encoding="utf-8"))
    source = Path(role["fit"]["path"]); fit = np.asarray(role["fit"]["sample_indices"], dtype=np.int64)
    val = np.asarray(role["validation"]["sample_indices"], dtype=np.int64)
    with h5py.File(str(source), "r") as h:
        y9 = np.asarray(h["known_label"][:, fit], dtype=np.uint8).T
    y = y9[:, [0, 1, 2, 3, 5, 7]]
    groups = defaultdict(list)
    for i, row in enumerate(y): groups[combo_key(row)].append(i)
    combo_names = sorted(groups)
    # Counts obtained by the audited ratio MILP preflight. They are fixed for
    # all seeds; only identities within each combination are randomized.
    plans = {
        "anchor_IF1": {"target_if": 1.0, "min_tail": None, "combo_counts": None},
        "IF3": {"target_if": 3.0, "min_tail": 500, "label_counts_reference": None},
        "IF5": {"target_if": 5.0, "min_tail": 500, "label_counts_reference": None},
        "IF6": {"target_if": 6.0, "min_tail": 500, "label_counts_reference": [5082, 2851, 4397, 4380, 4188, 847]},
        "IF10": {"target_if": 10.0, "min_tail": 500, "label_counts_reference": [5000, 4679, 4314, 3545, 4344, 500]},
        "IF20": {"target_if": 20.0, "min_tail": 200, "label_counts_reference": [4205, 3632, 4205, 4205, 4205, 210]},
    }
    # Derive combo counts for each reference label profile with a deterministic
    # integer program by importing the audited solver.
    from preflight_route_a_if_sweep_20260902 import solve_ratio
    for name, plan in plans.items():
        if name == "anchor_IF1": continue
        solved = solve_ratio(y, 8000, plan["target_if"], 2039, min_tail=plan["min_tail"])
        if solved is None: raise RuntimeError(f"no feasible plan for {name}")
        names, _, x, actual, h, t = solved
        plan.update({"combo_counts": {k: int(v) for k,v in zip(names,x) if int(v)>0},
                     "label_counts_reference": actual.tolist(), "head_label": NAMES[h], "tail_label": NAMES[t]})
    records=[]
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for name, plan in plans.items():
            if name == "anchor_IF1":
                local = np.arange(len(fit), dtype=np.int64); rng.shuffle(local); selected_local = np.sort(local[:8000])
            else:
                selected=[]
                for combo in combo_names:
                    count=int(plan["combo_counts"].get(combo,0))
                    cand=np.asarray(groups[combo],dtype=np.int64).copy(); rng.shuffle(cand); selected.extend(cand[:count].tolist())
                selected_local=np.asarray(selected,dtype=np.int64); rng.shuffle(selected_local)
            selected_global=np.sort(fit[selected_local])
            yy=y[selected_local] if name != "anchor_IF1" else y[selected_local]
            lc=yy.sum(0).astype(int); cc=defaultdict(int)
            for row in yy: cc[combo_key(row)]+=1
            payload={"schema":"route-a-if-sweep-manifest-v1","condition":name,"axis":"label_only",
                     "seed":int(seed),"source":str(source.resolve()),"source_fit_count":int(len(fit)),
                     "source_fit_indices_sha256":sha(fit),"selected_indices":selected_global.tolist(),
                     "selected_indices_sha256":sha(selected_global),"selected_count":int(len(selected_global)),
                     "requested_if":float(plan["target_if"]),"achieved_label_counts":lc.tolist(),
                     "effective_if":float(lc.max()/lc.min()),"combo_counts":dict(sorted(cc.items())),
                     "effective_cf":float(max(cc.values())/min(cc.values())),"unique_combo_count":int(len(cc)),
                     "validation_indices":val.tolist(),"validation_indices_sha256":sha(val),
                     "locked_test_used":False,"frozen_e5_modified":False,
                     "tail_min_constraint":plan["min_tail"],"head_label":plan.get("head_label"),"tail_label":plan.get("tail_label")}
            path=OUT/f"{name}__seed_{seed}.json"; path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); records.append(payload)
    summary={"schema":"route-a-if-sweep-manifest-summary-v1","status":"completed","fixed_train_count":8000,
             "conditions":{k:{"target_if":v["target_if"],"tail_min_constraint":v["min_tail"]} for k,v in plans.items()},
             "records":[{"condition":r["condition"],"seed":r["seed"],"selected_count":r["selected_count"],"effective_if":r["effective_if"],"effective_cf":r["effective_cf"],"unique_combo_count":r["unique_combo_count"],"label_counts":r["achieved_label_counts"]} for r in records],"locked_test_used":False,"frozen_e5_modified":False}
    (OUT.parent/"manifest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT.parent/"manifest_terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((OUT.parent/'manifest_summary.json').resolve()),"locked_test_used":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((OUT.parent/'manifest_summary.json').resolve()),"records":len(records)},ensure_ascii=False),flush=True)

if __name__ == "__main__": main()
