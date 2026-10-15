"""P3o-D: frozen time-frequency recognition and posterior-conditioned delay parameter residual."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3n_cross_interference_backbone import ISRJ_INDEX, output_parameter_metrics, sha256_file
from cj0724.p3o_low_power_attribution import MODEL_SEEDS,audit_identities,build_concat_dataset,close_concat_dataset,evaluate_outputs,fit_multi_file_normalizer,make_identity_roles,ordered_paths,role_metadata,save_prediction_payload,select_checkpoint_epochs,select_cp_safe_isrj_threshold
from cj0724.p3ob_dualview_representation import TemporalEncoder,collect_outputs,delay_peak_sequence,train_epoch
from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.train import get_device,set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr,optimize_class_thresholds

class FrozenFrequencyDelayResidual(nn.Module):
    def __init__(self, base: ParameterMatchedMultiView):
        super().__init__(); self.base=base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.delay=TemporalEncoder(3,10); self.residual=nn.Sequential(nn.LayerNorm(104),nn.Linear(104,12))
    def forward_with_embedding(self,iq):
        with torch.no_grad(): logits,base_parameter,embedding=self.base.forward_with_embedding(iq)
        delay=self.delay(delay_peak_sequence(iq)); raw=self.residual(torch.cat((embedding,delay),1)).reshape(-1,6,2)
        return logits,base_parameter+torch.sigmoid(logits).detach().unsqueeze(-1)*raw,embedding
    def forward(self,iq):
        logits,parameters,_=self.forward_with_embedding(iq); return logits,parameters
    def exclusive_contrast_loss(self,embedding,labels): return embedding.new_zeros(())

def _paths(project,tmpl): return {r:transition_files(project/tmpl.format(regime=r),'train',POWERS) for r in REGIMES}
def _outputs(model,loaders,device,norm,w): return {k:collect_outputs(model,v,device,norm,w) for k,v in loaders.items()}
def _class_nmae(record,name):
 rows=[x['normalized_mae'] for x in record['parameter']['per_target'] if x['class_name']==name]
 return float(np.mean(rows))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--train-root-template',default='generated_data/p3od_parameter_residual_train_seed1070000000_{regime}_rechunked'); p.add_argument('--development-root-template',default='generated_data/p3od_parameter_residual_dev_seed1080000000_{regime}_rechunked'); p.add_argument('--preregister',default='configs/p3od_frozen_frequency_delay_parameter_residual_preregister_20260801.json'); p.add_argument('--method-freeze',default='configs/p3od_frozen_frequency_delay_parameter_residual_method_freeze_20260802.json'); p.add_argument('--output-root',default='outputs/p3od_frozen_frequency_delay_parameter_residual_development_20260801_retry1'); p.add_argument('--run-root',default='outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156'); p.add_argument('--epochs',type=int,default=20);p.add_argument('--warmup-epochs',type=int,default=2);p.add_argument('--batch-size',type=int,default=63);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--parameter-weight',type=float,default=.1);p.add_argument('--contrast-weight',type=float,default=.0);p.add_argument('--gradient-clip-norm',type=float,default=5);p.add_argument('--cp-cap',type=float,default=.1);p.add_argument('--cp-confidence',type=float,default=.95);p.add_argument('--split-seed',type=int,default=20260801);p.add_argument('--identity-seeds',nargs=2,type=int,default=[1070000000,1080000000]);p.add_argument('--device',default='cuda');a=p.parse_args(); project=Path(__file__).resolve().parents[1]
 f=json.loads((project/a.method_freeze).read_text(encoding='utf-8'))
 if f['preregister_sha256']!=sha256_file(project/a.preregister) or f['implementation_sha256']!=sha256_file(Path(__file__).resolve()): raise RuntimeError('P3o-D frozen hash mismatch')
 if f['data_generator_sha256']!=sha256_file(project/'matlab/generate_p3o_low_power_attribution_data.m'): raise RuntimeError('P3o-D frozen generator mismatch')
 output=project/a.output_root; output.mkdir(parents=True,exist_ok=False); device=get_device(a.device); roots={'train':_paths(project,a.train_root_template),'development':_paths(project,a.development_root_template)}; refs={}; meta={'train':{},'development':{}}
 for role in roots:
  paired=None
  for r in REGIMES:
   audit_power_alignment(roots[role][r]); ids,labels=read_identity_metadata(roots[role][r][25]); paired=ids if paired is None else paired
   if not np.array_equal(paired,ids): raise RuntimeError('unpaired regimes')
   meta[role][r]={'identities':ids,'labels':labels}
  refs[role]=paired
 labels=meta['train']['ID_control']['labels']; devlabels=meta['development']['ID_control']['labels'];
 if not np.array_equal(labels,devlabels): raise RuntimeError('label mismatch')
 roles=make_identity_roles(labels,a.split_seed); old=json.loads((project/a.run_root/'protocol.json').read_text(encoding='utf-8')); audit=audit_identities(project,refs['train'],refs['development'],set(a.identity_seeds),Path(old['data_file']))
 if audit['cross_role_overlap'] or audit['maximum_prior_overlap']: raise RuntimeError('identity collision')
 (output/'identity_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');np.savez_compressed(output/'identity_roles.npz',**roles); (output/'protocol.json').write_text(json.dumps({'args':vars(a),'preregister_sha256':f['preregister_sha256'],'implementation_sha256':f['implementation_sha256'],'test_generated':False,'test_opened':False,'locked_test_used':False},indent=2),encoding='utf-8'); norm=fit_multi_file_normalizer(ordered_paths(roots['train']),roles['full_fit'],labels); datasets={x:build_concat_dataset(ordered_paths(roots['train']),roles[x]) for x in ('full_fit','early','calibration','selection')}; dev=build_concat_dataset(ordered_paths(roots['development']),np.arange(len(devlabels))); m={x:role_metadata(meta['train'],roles[x]) for x in ('early','calibration','selection')}; dm=role_metadata(meta['development'],np.arange(len(devlabels))); results=[]
 try:
  for seed in MODEL_SEEDS:
   set_seed(seed); fit=DataLoader(datasets['full_fit'],batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)); loaders={x:DataLoader(datasets[x],batch_size=a.batch_size) for x in ('early','calibration','selection')}; dl=DataLoader(dev,batch_size=a.batch_size); base=ParameterMatchedMultiView('F_time_frequency_matched').to(device); opt=torch.optim.AdamW(base.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); history=[]
   for e in range(1,a.epochs+1):
    _set_warmup_cosine_lr(opt,a.learning_rate,e,a.epochs,a.warmup_epochs); tr=train_epoch(base,fit,device,norm,opt,None,a); early=collect_outputs(base,loaders['early'],device,norm,a.parameter_weight); history.append({'epoch':e,'train':tr,'early_parameter_macro_nmae':float(output_parameter_metrics(early,norm)['oracle_macro_target_nmae']),'early_macro_aupr':float(average_precision_score(early['labels'].numpy(),torch.sigmoid(early['logits']).numpy(),average='macro'))}); torch.save(base.state_dict(),output/f'base_{seed}_{e}.pt')
   choice=select_checkpoint_epochs(history); epoch=int(choice['C_joint_epoch']);base.load_state_dict(torch.load(output/f'base_{seed}_{epoch}.pt',map_location=device,weights_only=True)); cal=collect_outputs(base,loaders['calibration'],device,norm,a.parameter_weight); th=optimize_class_thresholds(torch.sigmoid(cal['logits']).numpy(),cal['labels'].numpy()); safe=select_cp_safe_isrj_threshold(torch.sigmoid(cal['logits']).numpy(),cal['labels'].numpy(),m['calibration']['regime'],m['calibration']['power'],a.cp_cap,a.cp_confidence); th[ISRJ_INDEX]=safe['threshold']; base_dev=collect_outputs(base,dl,device,norm,a.parameter_weight)
   side=FrozenFrequencyDelayResidual(base).to(device); opt=torch.optim.AdamW(filter(lambda x:x.requires_grad,side.parameters()),lr=a.learning_rate,weight_decay=a.weight_decay); sh=[]
   for e in range(1,a.epochs+1):
    _set_warmup_cosine_lr(opt,a.learning_rate,e,a.epochs,a.warmup_epochs); tr=train_epoch(side,fit,device,norm,opt,None,a); early=collect_outputs(side,loaders['early'],device,norm,a.parameter_weight);sh.append({'epoch':e,'train':tr,'early_parameter_macro_nmae':float(output_parameter_metrics(early,norm)['oracle_macro_target_nmae']),'early_macro_aupr':0}); torch.save(side.state_dict(),output/f'side_{seed}_{e}.pt')
   se=min(sh,key=lambda x:x['early_parameter_macro_nmae'])['epoch']; side.load_state_dict(torch.load(output/f'side_{seed}_{se}.pt',map_location=device,weights_only=True)); side_dev=collect_outputs(side,dl,device,norm,a.parameter_weight); equal=bool(torch.equal(base_dev['logits'],side_dev['logits'])); results.append({'seed':seed,'base':evaluate_outputs(base_dev,th,dm,norm),'side':evaluate_outputs(side_dev,th,dm,norm),'logits_bitwise_equal':equal,'base_history':history,'side_history':sh,'base_epoch':epoch,'side_epoch':se,'safe_audit':safe}); save_prediction_payload(output/f'seed_{seed}_base.npz',base_dev,dm);save_prediction_payload(output/f'seed_{seed}_side.npz',side_dev,dm)
 finally:
  [close_concat_dataset(x) for x in datasets.values()];close_concat_dataset(dev)
 gates={'identity_overlap_zero':not audit['cross_role_overlap'] and not audit['maximum_prior_overlap'],'classification_logits_bitwise_equal_to_base':all(x['logits_bitwise_equal'] for x in results),'all_seeds_base_safe_calibration_feasible_and_development_maximum_cell_fpr_at_most_0p10':all(x['safe_audit']['feasible'] and x['base']['maximum_cell_fpr']<=.1 for x in results),'all_seeds_frequency_base_overall_isrj_recall_at_least_0p85':all(x['base']['isrj_recall']>=.85 for x in results),'all_seeds_frequency_base_25db_isrj_recall_at_least_0p78':all(x['base']['per_power']['25']['isrj_recall']>=.78 for x in results),'all_seeds_residual_oracle_and_predicted_gate_parameter_nmae_ratio_vs_base_at_most_0p98':all(x['side']['parameter']['oracle_macro_target_nmae']/x['base']['parameter']['oracle_macro_target_nmae']<=.98 and x['side']['predicted_gate']['predicted_gate_parameter_nmae']/x['base']['predicted_gate']['predicted_gate_parameter_nmae']<=.98 for x in results),'all_seeds_residual_miss_penalty_and_worst_combination_nonworse_vs_base':all(x['side']['predicted_gate']['miss_penalty_parameter_nmae']<=x['base']['predicted_gate']['miss_penalty_parameter_nmae'] and x['side']['predicted_gate']['worst_combo_predicted_gate_nmae']<=x['base']['predicted_gate']['worst_combo_predicted_gate_nmae'] for x in results),'all_seeds_residual_mft_and_isrj_oracle_nmae_ratio_vs_base_at_most_0p99':all(_class_nmae(x['side'],'MFT')/_class_nmae(x['base'],'MFT')<=.99 and _class_nmae(x['side'],'ISRJ')/_class_nmae(x['base'],'ISRJ')<=.99 for x in results),'test_generated_false':True,'test_opened_false':True,'locked_test_used_false':True};(output/'summary.json').write_text(json.dumps({'development_passed':all(gates.values()),'promotion_gates':gates,'results':results,'identity_audit':audit,'test_generated':False,'test_opened':False,'locked_test_used':False},default=lambda x:x.item() if hasattr(x,'item') else str(x),indent=2),encoding='utf-8')
if __name__=='__main__': main()
