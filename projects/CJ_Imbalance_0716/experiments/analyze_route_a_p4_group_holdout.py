"""Five-seed paired statistics for strict P4 group-holdout screening."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from experiments.run_route_a_nineclass_feature_sufficiency import sha256, write_json

def ci(x, seed=260904, draws=20000):
    x=np.asarray(x,dtype=float); r=np.random.default_rng(seed); s=x[r.integers(0,len(x),size=(draws,len(x)))].mean(1); return [float(np.quantile(s,.025)),float(np.quantile(s,.975))]
def run(args):
    seeds=args.seeds; a=[]; b=[]
    for s in seeds:
        a.append(json.loads((args.root/'F2_BCE'/f'F2_BCE__seed_{s}.json').read_text(encoding='utf-8'))['validation'])
        b.append(json.loads((args.root/'F2_BCE_counterfactualHN'/f'F2_BCE_counterfactualHN__seed_{s}.json').read_text(encoding='utf-8'))['validation'])
    out={'schema':'route-a-p4-group-holdout-statistics-v1','status':'completed','seeds':list(seeds),'locked_test_used':False,'comparisons':{}}
    for m in ('macro_aupr','macro_f1','combination_exact_accuracy'):
        d=np.asarray([x[m] for x in b])-np.asarray([x[m] for x in a]); w=wilcoxon(d,alternative='greater',method='auto')
        out['comparisons'][m]={'bce_values':[x[m] for x in a],'hn_values':[x[m] for x in b],'difference_values':d.tolist(),'mean_difference':float(d.mean()),'bootstrap_ci95':ci(d),'wilcoxon_greater_p':float(w.pvalue)}
    write_json(args.root/'p4_group_holdout_statistics.json',out); write_json(args.root/'p4_group_holdout_statistics_receipt.json',{'status':'completed','summary':str((args.root/'p4_group_holdout_statistics.json').resolve()),'locked_test_used':False,'script_sha256':sha256(Path(__file__))}); print(json.dumps({'status':'completed','summary':str(args.root/'p4_group_holdout_statistics.json')},ensure_ascii=False))
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p4_counterfactual_group_holdout_20260903'); p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]); return p.parse_args()
if __name__=='__main__': run(parse_args())
