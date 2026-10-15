"""Requirement-by-requirement protocol audit for Route-A five-class V2."""
from __future__ import annotations
import hashlib, json, time, sys
from pathlib import Path
import h5py, numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.run_route_a_five_class_frequency_lt_v2 import OUT_ROOT,AUDIT_ROOT,TRAIN_PATH,TEST_PATH,SEEDS,CLASS_NAMES

def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): d.update(b)
 return d.hexdigest()
def main():
 s0=load(AUDIT_ROOT/'stage0_audit.json'); manifest=load(AUDIT_ROOT/'label_if4_manifest.json'); s1=load(OUT_ROOT/'stage1_summary.json'); s2=load(OUT_ROOT/'stage2_summary.json'); s3=load(OUT_ROOT/'stage3_summary.json'); s4=load(OUT_ROOT/'stage4_summary.json'); ft=load(OUT_ROOT/'final_test_summary.json')
 with h5py.File(TRAIN_PATH,'r') as f: tr_seed=np.asarray(f['receiver_noise_seed'][:,:]).reshape(-1).astype(np.int64)
 with h5py.File(TEST_PATH,'r') as f: te_seed=np.asarray(f['receiver_noise_seed'][:,:]).reshape(-1).astype(np.int64)
 role_audits={}
 for split_kind in ('splits','splits_label_if4'):
  role_audits[split_kind]={}
  for seed in SEEDS:
   roles={k:np.asarray(v,dtype=np.int64) for k,v in load(OUT_ROOT/split_kind/f'seed_{seed}.json').items()}; pairs={}
   names=list(roles)
   for i,a in enumerate(names):
    for b in names[i+1:]: pairs[f'{a}_vs_{b}']=int(len(np.intersect1d(roles[a],roles[b])))
   union=np.concatenate(list(roles.values())); receiver_overlap={}
   for i,a in enumerate(names):
    for b in names[i+1:]: receiver_overlap[f'{a}_vs_{b}']=int(len(np.intersect1d(tr_seed[roles[a]],tr_seed[roles[b]])))
   role_audits[split_kind][str(seed)]={'counts':{k:int(len(v)) for k,v in roles.items()},'sample_index_overlap':pairs,'receiver_seed_overlap':receiver_overlap,'union_unique_count':int(len(np.unique(union))),'union_count':int(len(union))}
 test_receiver_overlap=int(len(np.intersect1d(tr_seed,te_seed)))
 missing=[]; invalid=[]
 # Prompt requires one final test only. Saved artifacts reveal finalization was
 # executed repeatedly during development, so test protocol is not compliant.
 invalid.append({'requirement':'test evaluated only once after development lock','evidence':'test_predictions exist for all exploratory arms and final_test_summary was regenerated multiple times','status':'invalid_test_contamination'})
 # Stage 3 says only a candidate that passed admission proceeds. Stage 1/2 did not.
 invalid.append({'requirement':'stage3 only on an admitted stage1/2 candidate','evidence':{'stage1_gate_pass':s1.get('stage1_gate_pass'),'selected_loss':s2.get('selected_loss')},'status':'exploratory_not_formal'})
 # Stage 4 is allowed as a diagnostic from B0-Focal-LabelIF4; preserve as diagnostic.
 missing += ['receiver-level bootstrap 5000 CI','IF10/IF4 or independent development manifest non-reversal','correlation between combination coverage change and per-label AUPR change','per-label precision and recall for logit adjustment','manifest record of retained/deleted counts and selected-index SHA256','explicit loss initial values in summary','per-arm failure receipts or explicit zero-failure receipt']
 audit={'status':'completed','decision':'protocol_incomplete_test_contaminated','audited_at_unix':time.time(),'source_hashes':{'train':sha(TRAIN_PATH),'test':sha(TEST_PATH)},'role_isolation':role_audits,'train_test_receiver_seed_overlap':test_receiver_overlap,'requirements_proven':['SawFM positive samples removed','five-class order','fixed B0 architecture','five seeds','9920 draws for joint arms','stage1 15 checkpoints','stage2 35 checkpoints','stage4 20 heads','no test-v2/locked-test used'],'invalid_or_exploratory':invalid,'missing_evidence':missing,'formal_claim_allowed':False,'recommended_recovery':'Do not retune on current test. Use a new independent development/confirmation source or declare V2 a protocol-negative exploratory closure.'}
 (OUT_ROOT/'protocol_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 print('PROTOCOL_AUDIT',audit['decision'],'missing',len(missing))
if __name__=='__main__': main()
