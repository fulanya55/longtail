"""Fill remaining V2 evidence using frozen saved artifacts only."""
from __future__ import annotations
import csv,json,sys,time,hashlib
from pathlib import Path
import h5py,numpy as np
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT_ROOT=ROOT/'outputs'/'route_a_five_class_frequency_lt_v2'
DATA_ROOT=Path('E:/LJQ_code')/''.join(map(chr,(0x5E72,0x6270,0x6570,0x636E)))/''.join(map(chr,(0x590D,0x5408,0x5E72,0x6270,0x6570,0x636E,0x30,0x37,0x31,0x36)))
AUDIT_ROOT=DATA_ROOT/'route_a_five_class_frequency_lt_v2'; TRAIN_PATH=DATA_ROOT/'closedset_6class_train.mat'; TEST_PATH=DATA_ROOT/'closedset_6class_test.mat'
CLASS_NAMES=('FMB','SinFM','MFT','VFT','ISRJ'); SEEDS=(272001,272002,272003,272004,272005)
STAGE2_ARMS=('Focal','DB','CB_BCE_beta0.99','CB_BCE_beta0.999','ASLDB_w0.50','ASLDB_w0.75','ASLDB_w0.90')

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def metrics(y,z): return np.array([average_precision_score(y[:,i],z[:,i]) for i in range(5)])
def fast_aupr(y, z):
 order=np.argsort(-z,axis=0); out=[]
 for i in range(y.shape[1]):
  yy=y[order[:,i],i].astype(np.float64); pos=yy.sum()
  out.append(float(np.sum((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy)/max(pos,1.0)))
 return np.asarray(out)
def main():
 s2=load(OUT_ROOT/'stage2_summary.json'); s3=load(OUT_ROOT/'stage3_summary.json'); ft=load(OUT_ROOT/'final_test_summary.json'); manifest=load(AUDIT_ROOT/'label_if4_manifest.json'); s0=load(AUDIT_ROOT/'stage0_audit.json')
 # Receiver bootstrap on locked saved test predictions for the preselected ASLDB0.50 vs Focal.
 with h5py.File(TEST_PATH,'r') as f: receiver=np.asarray(f['receiver_noise_seed'][:,:]).reshape(-1).astype(np.int64)
 test_idx=np.load(AUDIT_ROOT/'test_retained_indices.npy'); receiver=receiver[test_idx]; rng=np.random.RandomState(20260827); boot=[]; seed_point=[]
 for seed in SEEDS:
  base=np.load(OUT_ROOT/'stage2'/f'seed_{seed}'/'Focal'/'test_predictions.npz'); cand=np.load(OUT_ROOT/'stage2'/f'seed_{seed}'/'ASLDB_w0.50'/'test_predictions.npz'); y=base['targets']; zb=base['logits']; zc=cand['logits']; assert np.array_equal(y,cand['targets'])
  seed_point.append(float(metrics(y,zc).mean()-metrics(y,zb).mean()))
  ids=np.unique(receiver); position={int(v):i for i,v in enumerate(receiver.tolist())}; vals=[]
  # Each receiver_noise_seed is unique in this source, so this is an
  # identity/receiver bootstrap over saved predictions.
  for _ in range(1000):
   pos=rng.randint(0,len(ids),len(ids)); vals.append(float(fast_aupr(y[pos],zc[pos]).mean()-fast_aupr(y[pos],zb[pos]).mean()))
  boot.extend(vals)
 receiver_boot={'method':'receiver_noise_seed bootstrap over saved final test predictions; 1000 repetitions per model seed, pooled 5000','repetitions':5000,'per_seed_point_delta':seed_point,'mean':float(np.mean(seed_point)),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'positive_fraction':float(np.mean(np.array(boot)>0))}
 dump(OUT_ROOT/'receiver_bootstrap_summary.json',receiver_boot)

 # Logit adjustment precision/recall and threshold sensitivity.
 logit_rows=[]
 for item,testrow in zip(s3['rows'],ft['stage3_selected_rows']):
  seed=item['seed']; p=np.load(OUT_ROOT/'stage3'/f'seed_{seed}_selected.npz'); prior=p['train_prior']; tau=float(item['selected_tau']); raw=testrow['raw']; pred=np.load(OUT_ROOT/'stage2'/f'seed_{seed}'/'ASLDB_w0.50'/'test_predictions.npz'); y=pred['targets']; z=pred['logits']-tau*np.log((1-prior)/prior)[None,:]; prob=1/(1+np.exp(-np.clip(z,-50,50)))
  thresholds={}
  for t in (0.3,0.4,0.5,0.6,0.7): thresholds[str(t)]={'macro_precision':float(np.mean([precision_score(y[:,i],prob[:,i]>=t,zero_division=0) for i in range(5)])),'macro_recall':float(np.mean([recall_score(y[:,i],prob[:,i]>=t,zero_division=0) for i in range(5)])),'macro_f1':float(np.mean([f1_score(y[:,i],prob[:,i]>=t,zero_division=0) for i in range(5)]))}
  logit_rows.append({'seed':seed,'tau':tau,'raw_macro_aupr':raw['macro_aupr'],'adjusted_macro_aupr':testrow['adjusted']['macro_aupr'],'threshold_sensitivity':thresholds,'per_label_at_0.5':{CLASS_NAMES[i]:{'precision':float(precision_score(y[:,i],prob[:,i]>=.5,zero_division=0)),'recall':float(recall_score(y[:,i],prob[:,i]>=.5,zero_division=0))} for i in range(5)}})
 dump(OUT_ROOT/'logit_adjustment_diagnostics.json',{'rows':logit_rows,'mechanism_note':'per-label additive constants preserve within-label ranking; AUPR changes are numerical only'})

 # Coverage-change vs AUPR-change correlation (balanced Uniform vs LabelIF4 Focal).
 s1=load(OUT_ROOT/'stage1_summary.json'); label_if4=np.asarray(s0['label_if4_achieved_counts'],float); balanced=np.asarray(s0['train_counts_after_filter'],float); exposure_ratio=label_if4/balanced
 per_label=[]
 for i,n in enumerate(CLASS_NAMES):
  b=np.mean([r['test']['per_label'][n]['aupr'] for r in ft['joint_rows'] if r['stage']=='stage1' and r['arm']=='Uniform-Focal']); l=np.mean([r['test']['per_label'][n]['aupr'] for r in ft['joint_rows'] if r['stage']=='stage2' and r['arm']=='Focal']); per_label.append({'label':n,'positive_frequency_retention':float(exposure_ratio[i]),'aupr_delta_labelif4_minus_balanced':float(l-b)})
 correlation=float(np.corrcoef([x['positive_frequency_retention'] for x in per_label],[x['aupr_delta_labelif4_minus_balanced'] for x in per_label])[0,1])
 dump(OUT_ROOT/'coverage_performance_correlation.json',{'per_label':per_label,'pearson_frequency_retention_vs_aupr_delta':correlation,'limitation':'five labels only; descriptive, not causal'})

 # Initial losses and explicit zero-failure receipt.
 initial={}
 for arm in STAGE2_ARMS:
  initial[arm]=[{'seed':r['seed'],'epoch1_train_loss':load(r['history'])[0]['train_loss']} for r in s2['rows'] if r['arm']==arm]
 dump(OUT_ROOT/'loss_initial_values.json',initial)
 failures=[r for r in s1['rows']+s2['rows'] if r.get('status')=='failed']
 dump(OUT_ROOT/'failure_receipt.json',{'status':'completed','failed_run_count':len(failures),'failures':failures,'seed_replacement_used':False})

 # Enrich manifest evidence without rewriting the frozen manifest itself.
 selected=np.asarray(manifest['selected_indices'],dtype='<i8'); retained=np.load(AUDIT_ROOT/'train_retained_indices.npy').astype('<i8')
 dump(AUDIT_ROOT/'manifest_completeness_receipt.json',{'manifest_sha256':hashlib.sha256((AUDIT_ROOT/'label_if4_manifest.json').read_bytes()).hexdigest(),'source_sha256':manifest['source_sha256'],'retained_count':int(len(retained)),'deleted_count_from_filtered_pool':int(len(retained)-len(selected)),'selected_count':int(len(selected)),'selected_indices_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'retained_indices_sha256':hashlib.sha256(retained.tobytes()).hexdigest(),'class_counts':manifest['achieved_label_counts'],'combination_counts':manifest['combo_counts'],'construction_seed':manifest['seed'],'protocol_version':manifest['protocol_version'],'generated_at_unix':manifest['generated_at_unix']})

 # Gate 7 cannot be proven; create explicit preregistered non-execution receipt.
 dump(OUT_ROOT/'independent_manifest_gate_receipt.json',{'status':'not_executed','gate':'IF10/IF4 or independent development manifest non-reversal','reason':'No untouched independent five-class frequency-long-tail source exists in the authorized dataset. Current test is contaminated by repeated exploratory evaluation and cannot be reused for selection.','formal_gate_pass':False,'recommended_action':'Generate a new identity-isolated five-class source preserving all 31 non-empty combinations before any further model selection.'})
 dump(OUT_ROOT/'remaining_evidence_completion_receipt.json',{'status':'completed','generated_at_unix':time.time(),'artifacts':['receiver_bootstrap_summary.json','logit_adjustment_diagnostics.json','coverage_performance_correlation.json','loss_initial_values.json','failure_receipt.json','manifest_completeness_receipt.json','independent_manifest_gate_receipt.json'],'model_retrained':False,'candidate_reselected':False})
 print('REMAINING_EVIDENCE_OK')
if __name__=='__main__': main()
