"""Read-only 30-50 dB OOD atlas for the frozen P5b4 JNR attachment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, bootstrap_gain, metrics, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role, collect_frozen_posterior, with_context
from cj0724.p5b3_bounded_jnr_attachment import TARGET_CLASSES, TARGET_NAMES, collect_base_parameter, load_base, predicted_gated_role
from cj0724.p5b4_bounded_jnr_confirmation import load_head, estimate

POWERS=(30,35,40,45,50)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preregister',required=True);p.add_argument('--data-root',required=True);p.add_argument('--data-audit',required=True);p.add_argument('--output-root',required=True);a=p.parse_args(); project=Path.cwd()
    protocol=json.loads((project/a.preregister).read_text(encoding='utf-8')); audit=json.loads((project/a.data_audit).read_text(encoding='utf-8'))
    if not audit['valid_for_frozen_inference']: raise RuntimeError('P5c data audit failed')
    files=_power_files(project/a.data_root,POWERS); data=PairedPowerOracleDataset(files,np.arange(3780,dtype=np.int64)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); analytic=collect_analytic_role(data,device); rows=[]
    for source_seed,head_path in zip(protocol['frozen_candidate']['base_model_seeds'],protocol['frozen_candidate']['head_checkpoints']):
        model,normalizer,thresholds,source=load_base(project,int(source_seed));model=model.to(device).eval(); posterior=collect_frozen_posterior(data,model,device); role=with_context(analytic,posterior); head,state=load_head(project/head_path,device); value=estimate(role,head,state,device)
        logits,base_full=collect_base_parameter(data,model,normalizer,device); probability=1/(1+np.exp(-logits)); decision=probability>=thresholds[None,:]; base=base_full[:,TARGET_CLASSES,0].copy(); attached_full=base_full.copy()
        for local,cls in enumerate(TARGET_CLASSES): attached_full[decision[:,cls],cls,0]=value[decision[:,cls],local]
        attached=attached_full[:,TARGET_CLASSES,0]; allowed=np.zeros_like(base_full,bool)
        for cls in TARGET_CLASSES: allowed[:,cls,0]=decision[:,cls]
        scale=np.asarray(state['target_scale']); rows.append({'source':source,'head_sha256':sha256_file(project/head_path),'base':metrics(base,role,scale,powers=POWERS),'attached':metrics(attached,role,scale,powers=POWERS),'predicted_base':metrics(base,predicted_gated_role(role,probability,thresholds),scale,powers=POWERS),'predicted_attached':metrics(attached,predicted_gated_role(role,probability,thresholds),scale,powers=POWERS),'invariant':bool(np.array_equal(base_full[~allowed],attached_full[~allowed])),'base_prediction':base,'attached_prediction':attached,'role':role,'scale':scale})
        del model,head
        if device.type=='cuda':torch.cuda.empty_cache()
    atlas={}; bootstrap={}
    for local,name in enumerate(TARGET_NAMES):
        atlas[name]={};
        for power in POWERS:
            base_values=[r['base']['by_target'][name]['per_power_nmae'][str(power)] for r in rows]; new_values=[r['attached']['by_target'][name]['per_power_nmae'][str(power)] for r in rows]
            atlas[name][str(power)]={'base_nmae_by_model':base_values,'attached_nmae_by_model':new_values,'relative_gain_by_model':[1-n/o for n,o in zip(new_values,base_values)],'mean_relative_gain':float(np.mean([1-n/o for n,o in zip(new_values,base_values)]) )}
        bootstrap[name]=[bootstrap_gain(r['base_prediction'],r['attached_prediction'],r['role'],local,r['scale'],14041+i) for i,r in enumerate(rows)]
    result={'experiment':protocol['experiment'],'preregister_sha256':sha256_file(project/a.preregister),'data_audit':audit,'power_role':{'30':'in_range_anchor','35':'OOD','40':'OOD','45':'OOD','50':'OOD'},'rows':[ {k:v for k,v in r.items() if k not in ('base_prediction','attached_prediction','role','scale')} for r in rows],'atlas':atlas,'bootstrap':bootstrap,'classification_and_other_parameter_invariant':bool(all(r['invariant'] for r in rows)),'test_generated':False,'test_opened':False,'locked_test_used':False}
    out=project/a.output_root;out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'atlas':atlas,'invariant':result['classification_and_other_parameter_invariant']},indent=2));data.close()
if __name__=='__main__':main()
