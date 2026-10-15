"""P5d fixed-profile six-class LSCO comparison: shared MLCNN vs heterogeneous route."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from cj0724 import CLASS_NAMES
from cj0724.data import V73IQDataset,read_labels
from cj0724.multitask import PARAMETER_COLUMNS,build_multitask_model
from cj0724.multitask_benchmark import fit_normalizer
from cj0724.train import get_device,set_seed
from cj0724.tuned_multitask_sweep import PROFILE_CONFIGS,_collect_outputs,_metrics,_set_warmup_cosine_lr,_train_epoch,parameter_loss_target_weights

MODELS=('legacy_mlcnn_global','mlcnn_iq_grouped_task_routed'); SEEDS=(15041,15042,15043)
def class_audit(outputs):
 p=torch.sigmoid(outputs['logits']).numpy()>=.5;y=outputs['labels'].numpy().astype(bool)
 return {CLASS_NAMES[i]:{'recall':float((p[:,i]&y[:,i]).sum()/max(y[:,i].sum(),1)),'fpr':float((p[:,i]&~y[:,i]).sum()/max((~y[:,i]).sum(),1))} for i in range(6)}
def main():
 a=argparse.ArgumentParser();a.add_argument('--preregister',required=True);a.add_argument('--data-file',required=True);a.add_argument('--roles',required=True);a.add_argument('--role-receipt',required=True);a.add_argument('--output-root',required=True);x=a.parse_args();project=Path.cwd();proto=json.loads((project/x.preregister).read_text(encoding='utf-8')); receipt=json.loads((project/x.role_receipt).read_text(encoding='utf-8'))
 roles=np.load(project/x.roles);labels=read_labels(project/x.data_file);device=get_device('auto');out=project/x.output_root;out.mkdir(parents=True,exist_ok=True);results=[];profile=PROFILE_CONFIGS['cos_lr1e4_wd1e4']
 for fold in range(3):
  part={n:roles[f'fold{fold}_{n}'] for n in ('train','early','seen_confirm','unseen')}; normalizer=fit_normalizer(project/x.data_file,part['train'],labels)
  datasets={n:V73IQDataset(project/x.data_file,v,PARAMETER_COLUMNS) for n,v in part.items()}
  for model_name in MODELS:
   for seed in SEEDS:
    set_seed(seed);gen=torch.Generator().manual_seed(seed); loaders={n:DataLoader(d,batch_size=63,shuffle=n=='train',generator=gen if n=='train' else None,num_workers=0,pin_memory=device.type=='cuda') for n,d in datasets.items()}
    model=build_multitask_model(model_name,24,.2,256,64,'se').to(device);weights=parameter_loss_target_weights(model_name).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=True) if device.type=='cuda' else None;best=float('inf');state=None;best_epoch=0
    for epoch in range(1,21):
     _set_warmup_cosine_lr(opt,1e-4,epoch,20,2);_train_epoch(model,loaders['train'],device,normalizer,.1,opt,scaler,weights); early=_metrics(_collect_outputs(model,loaders['early'],device,normalizer,.1,weights),.5,normalizer);score=float(early['parameter']['oracle_macro_target_nmae'])
     if score<best:best,best_epoch,state=score,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    checkpoint=out/'checkpoints'/f'fold{fold}_{model_name}_seed{seed}_parameter.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({'model_state':state,'normalizer':normalizer.state_dict(),'best_epoch':best_epoch,'early_parameter_nmae':best},checkpoint)
    model.load_state_dict(state); payload={'fold':fold,'model':model_name,'seed':seed,'best_epoch':best_epoch,'early_parameter_nmae':best,'checkpoint':str(checkpoint),'seen':None,'unseen':None}
    for role,key in (('seen_confirm','seen'),('unseen','unseen')):
     outputs=_collect_outputs(model,loaders[role],device,normalizer,.1,weights);metric=_metrics(outputs,.5,normalizer);payload[key]={'metrics':metric,'per_class':class_audit(outputs),'logits_sha256':__import__('hashlib').sha256(outputs['logits'].numpy().tobytes()).hexdigest(),'sample_count':len(part[role])}
    results.append(payload);(out/'results.json').write_text(json.dumps({'preregister':str(project/x.preregister),'receipt':receipt,'results':results,'test_generated':False,'test_opened':False,'locked_test_used':False},indent=2),encoding='utf-8');del model,opt,loaders
    if device.type=='cuda':torch.cuda.empty_cache()
  for d in datasets.values():d.close()
 print(json.dumps({'runs':len(results),'output':str(out/'results.json')},indent=2))
if __name__=='__main__':main()
