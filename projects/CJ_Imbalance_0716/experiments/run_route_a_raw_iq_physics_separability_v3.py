"""Fixed raw-IQ physical feature separability screen on development-v3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels,read_parameters  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa:E402
from experiments.run_route_a_development_v2_h5 import _sha256,_write_json  # noqa:E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa:E402
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import (  # noqa:E402
    _load_roles,_matched_view,_rank_metrics,_scores,_train_probe,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES,TARGET_NAMES  # noqa:E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256,_validate_config  # noqa:E402
from experiments.run_shared import _data_path,load_config  # noqa:E402


FEATURE_NAMES=("MFT_ACF","VFT_STFT","ISRJ_Reset")


def _topk_mean(values,count=16):
    count=min(int(count),int(values.shape[-1])); return torch.topk(values,k=count,dim=-1).values.mean(dim=-1)


def _real_acf(values,maximum_lag,bins):
    length=int(values.shape[-1]); spectrum=torch.fft.rfft(values.float(),n=2*length,dim=-1)
    corr=torch.fft.irfft(spectrum*spectrum.conj(),n=2*length,dim=-1).abs()
    corr=corr[:,1:min(int(maximum_lag)+1,length)]
    corr=corr/corr.amax(dim=-1,keepdim=True).clamp_min(1e-8)
    avg=F.adaptive_avg_pool1d(corr[:,None],int(bins)).squeeze(1)
    maximum=F.adaptive_max_pool1d(corr[:,None],int(bins)).squeeze(1)
    return torch.cat([avg,maximum],dim=1)


def _mft_features(signal):
    z=torch.complex(signal[:,0].float(),signal[:,1].float()); length=int(z.shape[-1])
    spectrum=torch.fft.fft(z,n=2*length,dim=-1); corr=torch.fft.ifft(spectrum*spectrum.conj(),dim=-1).abs()
    corr=corr[:,1:min(1501,length)]; corr=corr/corr.amax(dim=-1,keepdim=True).clamp_min(1e-8)
    avg=F.adaptive_avg_pool1d(corr[:,None],64).squeeze(1)
    maximum=F.adaptive_max_pool1d(corr[:,None],64).squeeze(1)
    count=min(16,int(corr.shape[-1])); peaks,indices=torch.topk(corr,k=count,dim=-1)
    positions=indices.float()/float(max(int(corr.shape[-1])-1,1)); positions,order=positions.sort(dim=-1)
    peaks=peaks.gather(1,order)
    return torch.cat([avg,maximum,peaks,positions],dim=1)


def _vft_features(signal, grid_size=16):
    z=torch.complex(signal[:,0].float(),signal[:,1].float()); window=torch.hann_window(128,device=z.device)
    spectrum=torch.stft(z,n_fft=128,hop_length=32,window=window,center=True,
                        return_complex=True,onesided=False)
    power=torch.fft.fftshift(spectrum.abs().square(),dim=1)
    distribution=power/power.sum(dim=1,keepdim=True).clamp_min(1e-12)
    grid=int(grid_size)
    if grid <= 0:
        raise ValueError("grid_size must be positive")
    image=F.adaptive_avg_pool2d(torch.log1p(distribution*128.0)[:,None],(grid,grid)).flatten(1)
    coordinate=torch.linspace(-1.,1.,128,device=z.device,dtype=distribution.dtype)
    centroid=(distribution*coordinate[None,:,None]).sum(dim=1)
    spread=(distribution*(coordinate[None,:,None]-centroid[:,None,:]).square()).sum(dim=1).sqrt()
    peak=coordinate[distribution.argmax(dim=1)]
    tracks=torch.stack([centroid,spread,peak],dim=1)
    stats=torch.cat([tracks.mean(dim=-1),tracks.std(dim=-1,unbiased=False),
                     tracks.amax(dim=-1),torch.topk(tracks,k=min(8,int(tracks.shape[-1])),dim=-1).values.mean(dim=-1)],dim=1)
    return torch.cat([image,stats],dim=1)


def _isrj_features(signal):
    z=torch.complex(signal[:,0].float(),signal[:,1].float()); magnitude=z.abs()
    phase=torch.angle(z[:,1:]*z[:,:-1].conj())/torch.pi
    magnitude_mid=magnitude[:,1:]
    magnitude_delta=(magnitude[:,1:]-magnitude[:,:-1]).abs()
    phase_delta=(phase[:,1:]-phase[:,:-1]).abs()
    descriptors=(magnitude_mid[:,1:],magnitude_delta[:,1:],phase[:,1:].abs(),phase_delta)
    rows=[]
    for values in descriptors:
        stats=torch.stack([values.mean(dim=-1),values.std(dim=-1,unbiased=False),
                           values.amax(dim=-1),_topk_mean(values,16)],dim=1)
        rows.append(torch.cat([stats,_real_acf(values,512,32)],dim=1))
    return torch.cat(rows,dim=1)


def _extract(signals,device,batch_size):
    result={name:[] for name in FEATURE_NAMES}
    with torch.no_grad():
        for start in range(0,len(signals),int(batch_size)):
            batch=torch.from_numpy(signals[start:start+int(batch_size)]).to(device)
            for name,fn in zip(FEATURE_NAMES,(_mft_features,_vft_features,_isrj_features)):
                result[name].append(fn(batch).cpu().numpy().astype(np.float32))
    return {name:np.concatenate(rows) for name,rows in result.items()}


def _write_history(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _run_seed(args,config,labels6,weak6,seed,device,data_sha):
    seed_root=args.output_root/f"seed_{seed}";summary_path=seed_root/"seed_summary.json"
    if args.resume and summary_path.is_file():return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True,exist_ok=True)
    roles,role_indices_path=_load_roles(args.c6_root,seed,_data_path(config),config["normalization"],args.io_batch_size)
    features={role:_extract(roles[role]["signals"],device,args.feature_batch_size) for role in ("fit","validation","early")}
    prior=json.loads((args.layerwise_root/f"seed_{seed}"/"seed_summary.json").read_text(encoding="utf-8"))
    results={};counts={}
    for local,(target_index,target_name,feature_name) in enumerate(zip(TARGET_INDICES,TARGET_NAMES,FEATURE_NAMES)):
        mean=features["fit"][feature_name].mean(axis=0).astype(np.float32)
        std=np.maximum(features["fit"][feature_name].std(axis=0),1e-5).astype(np.float32)
        normalized={role:((features[role][feature_name]-mean[None,:])/std[None,:]).astype(np.float32) for role in features}
        views={};counts[target_name]={}
        for role in ("fit","validation","early"):
            selected,y,receipt=_matched_view(roles[role]["targets6"],weak6[roles[role]["sample_ids"]],target_index)
            views[role]=(normalized[role][selected],y);counts[target_name][role]=receipt
        train_seed=991000+int(seed)+local
        probe,history,info=_train_probe(*views["fit"],*views["validation"],train_seed,device,args)
        validation=_rank_metrics(views["validation"][1],_scores(probe,views["validation"][0],device))
        early=_rank_metrics(views["early"][1],_scores(probe,views["early"][0],device))
        baseline=prior["representations"]["T_FinalTemporal"][target_name]
        out=seed_root/target_name;out.mkdir(parents=True,exist_ok=True)
        torch.save({"model_state_dict":probe.state_dict(),"mean":mean,"std":std,
                    "feature_name":feature_name,"target":target_name,**info},out/"best_probe.pth")
        _write_history(out/"history.csv",history)
        results[target_name]={"feature_name":feature_name,"feature_dim":int(normalized["fit"].shape[1]),
                              "validation":validation,"early":early,
                              "validation_early_auroc_gap":validation["auroc"]-early["auroc"],
                              "final_temporal_early_auroc":baseline["early"]["auroc"],
                              "delta_vs_final_temporal":early["auroc"]-baseline["early"]["auroc"],**info}
        _write_json(out/"metrics.json",results[target_name])
    payload={"seed":int(seed),"status":"completed","smoke":bool(args.smoke),"data_sha256":data_sha,
             "role_indices_sha256":_sha256(role_indices_path),"labels":results,"matched_counts":counts,
             "fixed_features_no_learned_encoder":True,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    _write_json(summary_path,payload);return payload


def run(args):
    config=load_config(args.config);_validate_config(config);data_path=_data_path(config);data_sha=_sha256(data_path)
    if data_sha.lower()!=EXPECTED_V3_SHA256:raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:raise ValueError("test-v3 boundary is not sealed")
    labels6=read_labels(data_path);values,_=read_parameters(data_path)
    weak6=compute_weak_component_mask(labels6,_component_power(values),args.weak_threshold_db)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"));seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,labels6,weak6,seed,device,data_sha) for seed in seeds]
    aggregate={};gates={}
    for li,name in enumerate(TARGET_NAMES):
        aggregate[name]={metric:_mean_ci([row["labels"][name][metric] if metric in row["labels"][name] else row["labels"][name]["early"][metric] for row in rows],20273100+li*100+mi)
                         for mi,metric in enumerate(("auroc","aupr","validation_early_auroc_gap","delta_vs_final_temporal"))}
        positive=int(sum(row["labels"][name]["delta_vs_final_temporal"]>0 for row in rows))
        checks={} if args.smoke else {"absolute_ge_0p620":aggregate[name]["auroc"]["mean"]>=.620,
                                     "gain_ge_0p030":aggregate[name]["delta_vs_final_temporal"]["mean"]>=.030,
                                     "four_of_five_positive":positive>=4,
                                     "generalization_gap_lt_0p150":aggregate[name]["validation_early_auroc_gap"]["mean"]<.150}
        gates[name]={"positive_seeds":positive,"checks":checks,"passed":bool(checks and all(checks.values()))}
    payload={"status":"completed","experiment":"RouteA_RawIQPhysicsSeparability_v3","smoke":bool(args.smoke),
             "seeds":list(seeds),"data_sha256":data_sha,"aggregate":aggregate,
             "promotion_gate":{"labels":gates,"all_passed":bool(gates and all(row["passed"] for row in gates.values()))},
             "promoted_labels":[name for name,row in gates.items() if row["passed"]],
             "locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True);_write_json(args.output_root/"route_a_raw_iq_physics_separability_v3_summary.json",payload)
    print(json.dumps(payload,ensure_ascii=False,indent=2));return payload


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--layerwise-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_layerwise_matched_negative_separability_v3")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_raw_iq_physics_separability_v3")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]);parser.add_argument("--device",default=None)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--smoke",action="store_true");parser.add_argument("--smoke-epochs",type=int,default=2)
    parser.add_argument("--io-batch-size",type=int,default=256);parser.add_argument("--feature-batch-size",type=int,default=32)
    parser.add_argument("--batch-size",type=int,default=256);parser.add_argument("--learning-rate",type=float,default=1e-3)
    parser.add_argument("--epochs",type=int,default=80);parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--weight-decay",type=float,default=1e-4);parser.add_argument("--gradient-clip",type=float,default=5.0)
    parser.add_argument("--weak-threshold-db",type=float,default=10.0);return parser.parse_args()


if __name__=="__main__":run(parse_args())
