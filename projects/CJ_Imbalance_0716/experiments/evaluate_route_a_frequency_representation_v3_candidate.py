"""Evaluation-only pass for already frozen V3 candidate checkpoints."""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
SRC=ROOT/'experiments/run_route_a_frequency_representation_v3_candidate.py'
ns={'__file__':str(SRC),'__name__':'v3_candidate_module'}; exec(compile(SRC.read_text(encoding='utf-8'),str(SRC),'exec'),ns)
ResidualEvidence,cache_for,load_roles,load_manifest=ns['ResidualEvidence'],ns['cache_for'],ns['load_roles'],ns['load_manifest']
BASE=ns['BASE']; OUT=ns['OUT']; VH=ns['VH']; VL=ns['VL']; SEEDS=ns['SEEDS']; REGIMES=ns['REGIMES']
def main():
    device=torch.device('cuda'); roles=load_roles()
    for seed in SEEDS:
      for regime in REGIMES:
        selected=load_manifest(regime); opt=selected; early=roles['early']; ev=roles['evaluation']; indices=np.concatenate([opt,early,ev]); order={int(x):i for i,x in enumerate(indices)}; vidx=np.asarray([order[int(x)] for x in ev],dtype=np.int64)
        cache,_=cache_for(seed,regime,device,indices); st=torch.load(BASE/f'seed_{seed}'/regime/'F1_B0_DB.pt',map_location='cpu')['model_state_dict']; head=ResidualEvidence(st['classifier.1.weight'].numpy(),st['classifier.1.bias'].numpy()).to(device); cp=torch.load(OUT/f'seed_{seed}'/regime/'candidate.pt',map_location='cpu')['state_dict']; head.load_state_dict(cp); head.eval(); pred={}
        for view in ('VH','VL'):
          g,s,y=cache[view]
          with torch.no_grad(): logits=head(torch.as_tensor(g[vidx],device=device),torch.as_tensor(s[vidx],device=device))[0].cpu().numpy()
          pred[f'{view}_logits']=logits.astype(np.float32); pred[f'{view}_targets']=y[vidx].astype(np.uint8)
        p=OUT/f'seed_{seed}'/regime/'evaluation_predictions.npz'; np.savez_compressed(p,**pred); print(json.dumps({'seed':seed,'regime':regime,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
if __name__=='__main__': main()
