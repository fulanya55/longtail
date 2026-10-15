"""Aggregate IF-sweep evidence, drift, statistics and figures."""
from __future__ import annotations
import csv, json, math, sys, hashlib, subprocess
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); BASE=ROOT/'outputs/route_a_if_sweep_20260902'; RUNS=BASE/'runs'; TEST=BASE/'test_eval'; OUT=BASE/'final_summary'; OUT.mkdir(parents=True,exist_ok=True)
CONDS=['anchor_IF1','IF3','IF5','IF6','IF10','IF20','IF6_combo_matched']; NAMES=['FMB','AMA','SawFM','SinFM','MFT','ISRJ']; POWERS=[10,15,20,25,30,35,40,45,50]

def load_run(c): return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((RUNS/c).glob('*.json'))]
def load_test(c): return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(TEST.glob(c+'__seed_*.json'))]
def js_kl(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float); p=(p+1e-12)/(p.sum()+1e-12*len(p)); q=(q+1e-12)/(q.sum()+1e-12*len(q)); m=(p+q)/2
    klp=float(np.sum(p*np.log(p/m))); klq=float(np.sum(q*np.log(q/m))); return float((klp+klq)/2), float((klp+klq)/2)

def main():
    role=json.loads((ROOT/'outputs/route_a_c1_baseline_full_20260826/role_manifest.json').read_text(encoding='utf-8'))
    source=Path(role['fit']['path']); fit=np.asarray(role['fit']['sample_indices'],dtype=np.int64)
    import h5py
    from cjlt.data.split_builder import combo_key
    with h5py.File(str(source),'r') as h: y=np.asarray(h['known_label'][:,fit],dtype=np.uint8).T[:,[0,1,2,3,5,7]]
    combos=sorted({combo_key(r) for r in y}); base_counts=np.array([sum(combo_key(r)==k for r in y) for k in combos],float)
    hh=hashlib.sha256()
    with source.open('rb') as sf:
        for block in iter(lambda: sf.read(1<<20), b''): hh.update(block)
    source_sha=hh.hexdigest()
    try: gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    except Exception: gpu='unavailable'
    summary={'schema':'route-a-if-sweep-final-summary-v1','status':'completed','locked_test_used':False,'frozen_e5_modified':False,'source_fit_count':len(fit),'source_sha256':source_sha,'hardware':gpu,'conditions':{}}
    rows=[]
    for c in CONDS:
        rr=load_run(c); tt=load_test(c); allm=[]
        for r in rr:
            cc=np.array([r['manifest']['combo_counts'].get(k,0) for k in combos],float); js,sk=js_kl(cc,base_counts); allm.append({'run':r,'js':js,'sym_kl':sk})
        v={k:float(np.mean([x['run']['validation'][k] for x in allm])) for k in ['macro_aupr','macro_f1','combination_exact_accuracy','hamming_loss']}
        rec={'n_validation':len(rr),'n_test':len(tt),'effective_if_mean':float(np.mean([x['run']['manifest']['effective_if'] for x in allm])),'effective_cf_mean':float(np.mean([x['run']['manifest']['effective_cf'] for x in allm])),'unique_combo_mean':float(np.mean([x['run']['manifest']['unique_combo_count'] for x in allm])),'duplicate_count_max':int(max(x['run']['manifest']['selected_count']-len(set(x['run']['manifest']['selected_indices'])) for x in allm)),'js_drift_mean':float(np.mean([x['js'] for x in allm])),'sym_kl_drift_mean':float(np.mean([x['sym_kl'] for x in allm])),'elapsed_seconds_mean':float(np.mean([x['run'].get('elapsed_seconds',float('nan')) for x in allm])),'parameters':int(allm[0]['run'].get('total_parameters',0)),'validation_mean':v,'test_by_power':{}}
        for p in POWERS:
            vals=[x['test_metrics_by_power'][str(p)] for x in tt]; rec['test_by_power'][str(p)]={k:float(np.mean([m[k] for m in vals])) for k in ['macro_aupr','macro_f1','combination_exact_accuracy','hamming_loss']}
        summary['conditions'][c]=rec
        for x in allm:
            r=x['run']; rows.append({'condition':c,'seed':r['seed'],'effective_if':r['manifest']['effective_if'],'effective_cf':r['manifest']['effective_cf'],'selected_count':r['manifest']['selected_count'],'js_drift':x['js'],'sym_kl_drift':x['sym_kl'],'val_macro_aupr':r['validation']['macro_aupr'],'val_macro_f1':r['validation']['macro_f1'],'val_combo':r['validation']['combination_exact_accuracy'],'val_hamming':r['validation']['hamming_loss']})
    # paired IF6 vs anchor validation on common three seeds
    A={r['seed']:r for r in load_run('anchor_IF1')}; I={r['seed']:r for r in load_run('IF6')}
    summary['paired_anchor_vs_IF6_validation']={}
    rng=np.random.default_rng(20260902)
    for k in ['macro_aupr','macro_f1','combination_exact_accuracy','hamming_loss']:
        d=np.array([I[s]['validation'][k]-A[s]['validation'][k] for s in [2039,2040,2041]],float); boots=np.asarray([rng.choice(d,size=len(d),replace=True).mean() for _ in range(20000)]); summary['paired_anchor_vs_IF6_validation'][k]={'delta_mean':float(d.mean()),'wilcoxon_p':float(wilcoxon(d).pvalue),'deltas':d.tolist(),'bootstrap95_ci':[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'run_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Curves
    x=np.arange(len(CONDS),dtype=float); labels=['IF≈1','IF=3','IF=5','IF=6','IF=10','IF≈20','IF6+63combos']
    fig,ax=plt.subplots(1,3,figsize=(12,3.7),dpi=180)
    for j,(key,title) in enumerate([('validation_mean','Validation Macro-F1'),('validation_mean','Validation combo accuracy'),('test_by_power','30 dB test combo accuracy')]):
        if j<2: yv=[summary['conditions'][c]['validation_mean']['macro_f1' if j==0 else 'combination_exact_accuracy'] for c in CONDS]
        else: yv=[summary['conditions'][c]['test_by_power']['30']['combination_exact_accuracy'] for c in CONDS]
        ax[j].plot(x,yv,'o-',color='#2457a6'); ax[j].set_xticks(x,labels); ax[j].set_ylim(0,1.03); ax[j].set_title(title); ax[j].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'if_effect_curves.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.7),dpi=180); vals=[summary['conditions'][c]['js_drift_mean'] for c in CONDS]; ax.bar(labels,vals,color='#b85c36'); ax.set_ylabel('JS drift vs original fit'); ax.set_title('Combination-distribution drift'); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'combination_drift.png',bbox_inches='tight'); plt.close(fig)
    (OUT/'terminal_receipt.json').write_text(json.dumps({'status':'completed','summary':str((OUT/'summary.json').resolve()),'locked_test_used':False,'frozen_e5_modified':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','summary':str((OUT/'summary.json').resolve()),'rows':len(rows)},ensure_ascii=False))
if __name__=='__main__': main()
