"""Read-only diagnosis of VFT frozen-threshold transport on existing ISRJ OOD."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
PROJECT_ROOT=Path(__file__).resolve().parents[1]
VFT=4

def _oracle_threshold(probability, targets, max_fpr=.20):
    negative=np.asarray(targets,dtype=np.uint8)==0;values=np.asarray(probability,dtype=np.float64)[negative]
    allowed=int(np.floor(float(max_fpr)*len(values)))
    if allowed<=0:return float(np.nextafter(values.max(),np.inf))
    ordered=np.sort(values)[::-1]
    return float(np.nextafter(ordered[min(allowed-1,len(ordered)-1)],np.inf))

def _row(path,seed,max_fpr):
    item=np.load(path);targets=np.asarray(item["targets"],dtype=np.uint8);weak=np.asarray(item["weak_mask"],dtype=bool);prob=np.asarray(item["probabilities"],dtype=np.float64)[:,VFT];threshold=float(np.asarray(item["thresholds"])[VFT]);positive=targets[:,VFT]==1;negative=~positive;weak_positive=positive & weak[:,VFT]
    oracle=_oracle_threshold(prob,targets[:,VFT],max_fpr);rank=weak_positive|negative
    def metrics(value):
        prediction=prob>=value
        return {"threshold":float(value),"negative_fpr":float(prediction[negative].mean()),"weak_recall":float(prediction[weak_positive].mean()),"weak_vs_negative_auroc":float(roc_auc_score(weak_positive[rank].astype(np.uint8),prob[rank])),"weak_positive_count":int(weak_positive.sum()),"negative_count":int(negative.sum())}
    return {"seed":int(seed),"frozen":metrics(threshold),"oracle_fpr20_diagnostic":metrics(oracle),"oracle_minus_frozen_threshold":float(oracle-threshold)}

def _mean(rows,path):return float(np.mean([next_value(row,path) for row in rows]))
def next_value(row,path):
    value=row
    for key in path:value=value[key]
    return value

def build(root=PROJECT_ROOT,smoke=False,max_fpr=.20):
    source=root/"outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen";seeds=[2039,2040,2041,2042,2043];seeds=seeds[:1] if smoke else seeds
    rows=[_row(source/f"seed_{seed}"/"Q4_VFTAlignedRidgeFusion"/"predictions.npz",seed,max_fpr) for seed in seeds]
    frozen_fpr=_mean(rows,("frozen","negative_fpr"));oracle_fpr=_mean(rows,("oracle_fpr20_diagnostic","negative_fpr"));frozen_weak=_mean(rows,("frozen","weak_recall"));oracle_weak=_mean(rows,("oracle_fpr20_diagnostic","weak_recall"));auroc=_mean(rows,("frozen","weak_vs_negative_auroc"))
    diagnosis={"frozen_fpr_exceeds_target":frozen_fpr>max_fpr,"ranking_insufficient_mean_auroc_lt_0p65":auroc<.65,"oracle_weak_recall_loss":oracle_weak-frozen_weak,"oracle_fpr_control_requires_weak_recall_loss_gt_0p03":oracle_weak-frozen_weak<-.03}
    diagnosis["threshold_only_correction_not_supported"]=bool(diagnosis["ranking_insufficient_mean_auroc_lt_0p65"] or diagnosis["oracle_fpr_control_requires_weak_recall_loss_gt_0p03"])
    return {"status":"completed","experiment":"RouteA_VFTOODThresholdTransportDiagnostic","smoke":bool(smoke),"evidence_role":"read_only_existing_external_ood_diagnostic","seeds":seeds,"max_negative_fpr":max_fpr,"rows":rows,"aggregate":{"frozen_fpr":frozen_fpr,"oracle_fpr":oracle_fpr,"frozen_weak_recall":frozen_weak,"oracle_weak_recall":oracle_weak,"frozen_weak_vs_negative_auroc":auroc,"mean_oracle_threshold_shift":_mean(rows,("oracle_minus_frozen_threshold",))},"diagnosis":diagnosis,"external_refit":False,"hyperparameter_search":False,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_vft_ood_threshold_transport_diagnostic/route_a_vft_ood_threshold_transport_diagnostic_summary.json");p.add_argument("--smoke",action="store_true");p.add_argument("--max-negative-fpr",type=float,default=.20);args=p.parse_args();payload=build(smoke=args.smoke,max_fpr=args.max_negative_fpr);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
