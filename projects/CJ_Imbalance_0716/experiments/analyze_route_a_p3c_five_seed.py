"""Statistical audit for the five-seed P3c development run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, sha256, write_json

def ci(values, seed=260831, draws=20000):
    x=np.asarray(values,dtype=float); rng=np.random.default_rng(seed); sample=x[rng.integers(0,len(x),size=(draws,len(x)))].mean(axis=1); return [float(np.quantile(sample,.025)),float(np.quantile(sample,.975))]

def run(args):
    roots=[args.output_root/'F2_BCE_30dB',args.output_root/'F2_BCE_power_consistency']; seeds=args.seeds; rows={name:[] for name in ('bce','consistency')}
    for key,root in zip(rows,roots):
        for seed in seeds: rows[key].append(json.loads((root/f'{root.name}__seed_{seed}.json').read_text(encoding='utf-8')))
    def vals(key, field, power=None): return np.asarray([r['validation'][field] if power is None else r['power_dev'][str(power)][field] for r in rows[key]],dtype=float)
    out={'schema':'route-a-p3c-five-seed-statistics-v1','status':'completed','seeds':list(seeds),'locked_test_used':False,'comparisons':{}}
    for metric in ('macro_aupr','macro_f1','combination_exact_accuracy'):
        d=vals('consistency',metric) - vals('bce',metric)
        try: w=wilcoxon(d,alternative='greater',method='exact')
        except Exception: w=wilcoxon(d,alternative='greater')
        out['comparisons']['validation_'+metric]={'bce_values':vals('bce',metric).tolist(),'consistency_values':vals('consistency',metric).tolist(),'paired_difference_values':d.tolist(),'mean_difference':float(d.mean()),'bootstrap_ci95':ci(d),'wilcoxon_greater_p':float(w.pvalue)}
    for p in POWERS_DB:
        d=vals('consistency','macro_aupr',p)-vals('bce','macro_aupr',p); dc=vals('consistency','combination_exact_accuracy',p)-vals('bce','combination_exact_accuracy',p)
        out['comparisons'][f'power_{p}']={'macro_aupr_difference_values':d.tolist(),'macro_aupr_mean_difference':float(d.mean()),'macro_aupr_bootstrap_ci95':ci(d,seed=260831+p),'macro_aupr_wilcoxon_greater_p':float(wilcoxon(d,alternative='greater',method='auto').pvalue) if np.any(d) else 1.0,'combo_difference_values':dc.tolist(),'combo_mean_difference':float(dc.mean()),'combo_bootstrap_ci95':ci(dc,seed=260900+p)}
    write_json(args.output_root/'p3c_five_seed_statistics.json',out); write_json(args.output_root/'p3c_statistics_receipt.json',{'status':'completed','summary':str((args.output_root/'p3c_five_seed_statistics.json').resolve()),'locked_test_used':False,'script_sha256':sha256(Path(__file__))}); print(json.dumps({'status':'completed','summary':str(args.output_root/'p3c_five_seed_statistics.json')},ensure_ascii=False))

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p3c_power_consistency_5seed_20260829'); p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043]); return p.parse_args()
if __name__=='__main__': run(parse_args())
