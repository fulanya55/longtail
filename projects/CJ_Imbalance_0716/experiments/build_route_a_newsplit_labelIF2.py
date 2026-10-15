"""Build a fresh identity split and audited label-IF2 manifests."""
from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path
import sys,h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.data.longtail_manifest import build_label_longtail
from cjlt.data.split_builder import combo_key
from experiments.run_route_a_c1_baseline import CLASS_NAMES,SELECTED_INDICES,pure_six_indices

def data_root():
    parts=["".join(chr(c) for c in [0x8bba,0x6587,0x76f8,0x5173]),"".join(chr(c) for c in [0x5e72,0x6270,0x6570,0x636e]),"".join(chr(c) for c in [0x590d,0x5408,0x5e72,0x6270,0x6570,0x636e])+"0716","".join(chr(c) for c in [0x4e5d,0x7c7b])]
    return Path("E:/").joinpath(*parts)
def sha(a): return hashlib.sha256(np.asarray(a,dtype="<i8").tobytes()).hexdigest()
def main():
    out=ROOT/"outputs/route_a_newsplit_labelIF2_20260901";out.mkdir(parents=True,exist_ok=True)
    source=next(x for x in data_root().rglob("closedset_9class_train.mat") if "level_05" in str(x));pure=pure_six_indices(source);perm=np.random.default_rng(260901).permutation(pure);fit=np.sort(perm[:4096]);val=np.sort(perm[4096:6144])
    with h5py.File(str(source),"r") as h: yp=np.asarray(h["known_label"][:,pure],dtype=np.uint8).T
    labels=yp[:,list(SELECTED_INDICES)]; fit_local=np.searchsorted(pure,fit); yf=labels[fit_local]
    role={"schema":"route-a-newsplit-labelIF2-role-v1","selected_classes":list(CLASS_NAMES),"source":str(source.resolve()),"split_seed":260901,"fit_count":len(fit),"validation_count":len(val),"fit_indices_sha256":sha(fit),"validation_indices_sha256":sha(val),"locked_test_used":False,"test_files_opened":False,"frozen_e5_modified":False}
    (out/"role_manifest.json").write_text(json.dumps(role,ensure_ascii=False,indent=2),encoding="utf-8")
    targets=[800,696,606,528,459,400];records=[]
    for seed in (2039,2040,2041,2042,2043):
        m=build_label_longtail(yf,np.arange(len(yf)),targets,seed,tolerance=.05,source_split_sha256=sha(fit),target_sample_count=1600,sample_count_tolerance=0)
        selected=fit[m.selected_indices]; ysel=labels[np.searchsorted(pure,selected)]; counts=ysel.sum(0).astype(int); cc=Counter(combo_key(row) for row in ysel)
        payload={"schema":"route-a-newsplit-labelIF2-manifest-v1","condition":"new_label_IF2","seed":seed,"selected_indices":selected.astype(int).tolist(),"selected_count":len(selected),"requested_if":2.0,"achieved_label_counts":counts.tolist(),"effective_if":float(counts.max()/counts.min()),"combo_counts":{str(k):int(v) for k,v in sorted(cc.items())},"selected_indices_sha256":sha(selected),"validation_indices":val.astype(int).tolist(),"validation_count":len(val),"validation_indices_sha256":sha(val),"locked_test_used":False}
        (out/f"new_label_IF2__seed_{seed}.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");records.append(payload)
    summary={"schema":"route-a-newsplit-labelIF2-summary-v1","status":"completed","locked_test_used":False,"frozen_e5_modified":False,"fit_count":len(fit),"validation_count":len(val),"records":[{"seed":r["seed"],"selected_count":r["selected_count"],"effective_if":r["effective_if"],"effective_cf":max(r["combo_counts"].values())/min(r["combo_counts"].values())} for r in records]}
    (out/"manifest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(out/"terminal_receipt.json").write_text(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"locked_test_used":False,"test_files_opened":False},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","summary":str((out/"manifest_summary.json").resolve()),"records":summary["records"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
