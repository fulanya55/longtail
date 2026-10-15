"""Aggregate V3 candidate against per-cell best frozen baseline and bootstrap."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'outputs/route_a_frequency_representation_v3/stage2_formal'; CAND=ROOT/'outputs/route_a_frequency_representation_v3/stage3_candidate'; SEEDS=(261021,261022,261023,261024,261025); REGIMES=('FB','LabelIF20','ComboIF20'); CLS=('FMB','SawFM','SinFM','MFT','VFT','ISRJ')
def ap(y,z): return np.asarray([average_precision_score(y[:,j],z[:,j]) for j in range(6)],dtype=float)
def load(seed,reg):
    s=json.loads((BASE/f'seed_{seed}'/reg/'summary.json').read_text()); best=max(('F0_B0_Focal','F1_B0_DB','F2_Ordinary_cRT','F3_RBC_cRT'),key=lambda a:s['results'][a]['by_view']['VH']['macro_aupr']); b=np.load(BASE/f'seed_{seed}'/reg/'evaluation_predictions.npz'); c=np.load(CAND/f'seed_{seed}'/reg/'evaluation_predictions.npz'); return s,b,c,best
def main():
    rows=[]; arrays={}
    for seed in SEEDS:
      for reg in REGIMES:
        s,b,c,best=load(seed,reg); y=b['targets']; arrays[(seed,reg)]={'y':y,'base':b[best+'_VH_logits'],'cand':c['VH_logits']}; by={}
        for view in ('VH','VL'):
          base=b[best+'_'+view+'_logits']; cand=c[view+'_logits']; mb=ap(y,base); mc=ap(y,cand); by[view]={'baseline_arm':best,'baseline_macro':float(mb.mean()),'candidate_macro':float(mc.mean()),'macro_delta':float((mc-mb).mean()),'per_label_delta':{CLS[j]:float(mc[j]-mb[j]) for j in range(6)}}
        rows.append({'seed':seed,'regime':reg,'best_baseline_arm':best,'VH':by['VH'],'VL':by['VL']})
    primary=[r for r in rows if r['regime'] in ('LabelIF20','ComboIF20')]; if20=[r['VH']['macro_delta'] for r in rows if r['regime']=='LabelIF20']; combo=[r['VH']['macro_delta'] for r in rows if r['regime']=='ComboIF20']; fb=[r['VH']['macro_delta'] for r in rows if r['regime']=='FB']; tail=[np.mean([r['VH']['per_label_delta'][x] for x in ('SawFM','ISRJ','VFT')]) for r in primary]
    out={'status':'completed','rows':rows,'summary':{'labelif20_vh_mean':float(np.mean(if20)),'comboif20_vh_mean':float(np.mean(combo)),'fb_vh_mean':float(np.mean(fb)),'primary_tail_vh_mean':float(np.mean(tail)),'labelif20_positive_count':int(sum(x>0 for x in if20)),'comboif20_positive_count':int(sum(x>0 for x in combo))},'locked_test_used':False,'test_v2_used':False,'test_v3_used':False}
    (CAND/'candidate_point_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
