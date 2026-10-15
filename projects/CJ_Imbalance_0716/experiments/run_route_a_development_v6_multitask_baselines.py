"""Development-v6 fair comparison with two-task multitask optimizers."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa:E402
from cjlt.data.v73_dataset import read_labels,read_parameters  # noqa:E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa:E402
from experiments.run_route_a_development_v2_h5 import COMMON_INDICES,COMMON_NAMES,TARGET_NAMES,_indices_sha256,_read_json,_sha256,_target_weak_rank,_write_history,_write_json  # noqa:E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME,_component_power,_configure_partial_scope,_loss_components,_macro_aupr,_predict  # noqa:E402
from experiments.run_route_a_development_v4_sawfm_specificity import _apply_direction,_build_roles,_direction_norm,_evaluate_model_v4,_load_model_from_state,_metric,_save_and_evaluate,_state_direction,_train_endpoint,_variant_map  # noqa:E402
from experiments.run_route_a_dfms_iq_frozen_probe import _load_backbone  # noqa:E402
from experiments.run_shared import _data_path,load_config  # noqa:E402

VARIANTS=("C6_Joint","M5_GradientIsolated","LossWeight0p1","ModelAverage0p5","PCGradTwoTask","MGDATwoTask","CAGrad0p4TwoTask","GradNormTwoTask","FullAlpha1p5")
TRAINED_BASELINES=("LossWeight0p1","PCGradTwoTask","MGDATwoTask","CAGrad0p4TwoTask","GradNormTwoTask")

def _validate_v6_config(config):
    if bool(config.get('allow_locked_test',False)): raise ValueError('development-v6 requires allow_locked_test=false')
    if str(config.get('evidence_role'))!='development_v6': raise ValueError('evidence_role must be development_v6')
    if bool(config.get('test_v6_generated',True)) or bool(config.get('test_v6_access_authorized',True)) or bool(config.get('test_v6_used',True)): raise ValueError('test-v6 must remain absent and unused')
    if not bool(config.get('test_v3_sealed',False)) or bool(config.get('test_v3_used',True)): raise ValueError('test-v3 must remain sealed and unused')
    if not bool(config.get('test_v2_consumed',False)): raise ValueError('test-v2 must remain consumed')
    if bool(config.get('development_v5_reuse_for_selection',True)): raise ValueError('development-v5 reuse for selection must be false')
    data=(Path(str(config['data_root']))/str(config['train_file'])).resolve()
    forbidden={Path(str(config['test_v6_path'])).resolve(),Path(str(config['sealed_test_v3_path'])).resolve(),Path(str(config['forbidden_test_v2_path'])).resolve()}
    if data in forbidden or 'test' in data.name.lower(): raise ValueError('development-v6 data resolves to a test role')

def _grad_list(loss,parameters,create_graph=False):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,create_graph=create_graph,allow_unused=True)
    return [torch.zeros_like(parameter) if gradient is None else gradient for parameter,gradient in zip(parameters,gradients)]

def _dot(first,second): return sum((left*right).sum() for left,right in zip(first,second))
def _norm(values): return torch.sqrt(sum((value*value).sum() for value in values)+1e-24)

def _pcgrad_two(first,second):
    dot=_dot(first,second)
    left=[value.clone() for value in first]; right=[value.clone() for value in second]
    projected=bool(float(dot.detach().cpu())<0.0)
    if projected:
        left=[value-dot/(_dot(second,second)+1e-12)*basis for value,basis in zip(left,second)]
        right=[value-dot/(_dot(first,first)+1e-12)*basis for value,basis in zip(right,first)]
    return [a+b for a,b in zip(left,right)],projected

def _mgda_weight(first,second):
    difference=[a-b for a,b in zip(first,second)]
    denominator=float(_dot(difference,difference).detach().cpu())
    if denominator<=1e-24: return 0.5
    numerator=float(sum(((b-a)*b).sum() for a,b in zip(first,second)).detach().cpu())
    return float(np.clip(numerator/denominator,0.0,1.0))

def _cagrad_weight(first,second,c=0.4,grid_size=101):
    a11=float(_dot(first,first).detach().cpu()); a12=float(_dot(first,second).detach().cpu()); a22=float(_dot(second,second).detach().cpu())
    matrix=np.asarray([[a11,a12],[a12,a22]],dtype=np.float64); base=np.asarray([0.5,0.5])
    g0_norm=float(np.sqrt(max(base@matrix@base,0.0))); coefficient=float(c)*g0_norm
    best=None
    for value in np.linspace(0.0,1.0,int(grid_size)):
        weight=np.asarray([value,1.0-value]); objective=float(weight@matrix@base+coefficient*np.sqrt(max(weight@matrix@weight,0.0)+1e-24))
        candidate=(objective,float(value))
        if best is None or candidate<best: best=candidate
    return best[1]

def _assign_gradients(parameters,gradients):
    for parameter,gradient in zip(parameters,gradients): parameter.grad=gradient.detach().clone()

def _train_baseline(model,variant,roles,weak6,device,seed,args,output_dir):
    scope=_configure_partial_scope(model); parameters=scope['shared']+scope['classifier']
    positives=roles['fit']['targets6'].sum(axis=0).astype(np.float32)
    pos_weight=torch.as_tensor((len(roles['fit']['targets6'])-positives)/np.maximum(positives,1.0),dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW([{"params":scope['shared'],"lr":float(args.learning_rate)*float(args.representation_lr_scale)},{"params":scope['classifier'],"lr":float(args.learning_rate)}],weight_decay=float(args.weight_decay))
    task_weights=None; weight_optimizer=None; initial_losses=None
    if variant=='GradNormTwoTask':
        task_weights=torch.nn.Parameter(torch.ones(2,device=device)); weight_optimizer=torch.optim.Adam([task_weights],lr=float(args.gradnorm_weight_lr))
    train_seed=int(seed)+73000; torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(roles['fit']['signals']),torch.from_numpy(roles['fit']['targets6'].astype(np.float32))),batch_size=int(args.batch_size),shuffle=True,generator=torch.Generator().manual_seed(train_seed),num_workers=0)
    best_state=None; best_key=(-float('inf'),-float('inf')); best_epoch=0; stale=0; history=[]; started=time.time()
    cosine_rows=[]; conflict_count=0; coefficient_rows=[]; clipping_count=0
    for epoch in range(1,int(args.epochs)+1):
        model.train(); losses_epoch=[]
        for x,y in loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x); common,saw=_loss_components(logits,y,pos_weight); tasks=[common,0.2*saw]
            if variant=='LossWeight0p1':
                total=common+0.1*saw; total.backward()
            elif variant in ('PCGradTwoTask','MGDATwoTask','CAGrad0p4TwoTask'):
                first=_grad_list(tasks[0],parameters); second=_grad_list(tasks[1],parameters)
                cosine=float((_dot(first,second)/(_norm(first)*_norm(second)+1e-12)).detach().cpu()); cosine_rows.append(cosine); conflict_count+=int(cosine<0)
                if variant=='PCGradTwoTask': combined,projected=_pcgrad_two(first,second); coefficient_rows.append(float(projected))
                elif variant=='MGDATwoTask':
                    weight=_mgda_weight(first,second); coefficient_rows.append(weight); combined=[2.0*(weight*a+(1.0-weight)*b) for a,b in zip(first,second)]
                else:
                    weight=_cagrad_weight(first,second,c=float(args.cagrad_c),grid_size=int(args.cagrad_grid)); coefficient_rows.append(weight)
                    g0=[0.5*(a+b) for a,b in zip(first,second)]; gw=[weight*a+(1.0-weight)*b for a,b in zip(first,second)]
                    lam=float(args.cagrad_c)*float(_norm(g0).detach().cpu())/(float(_norm(gw).detach().cpu())+1e-12)
                    combined=[2.0*(base+lam*weighted)/(1.0+float(args.cagrad_c)**2) for base,weighted in zip(g0,gw)]
                _assign_gradients(parameters,combined); total=tasks[0]+tasks[1]
            elif variant=='GradNormTwoTask':
                if initial_losses is None: initial_losses=torch.stack([loss.detach() for loss in tasks])
                weighted=[task_weights[index]*tasks[index] for index in range(2)]
                gradient_norms=[]
                for value in weighted:
                    gradients=_grad_list(value,scope['shared'],create_graph=True); gradient_norms.append(_norm(gradients))
                gradient_norms=torch.stack(gradient_norms)
                rate=torch.stack([loss.detach() for loss in tasks])/initial_losses; inverse=rate/(rate.mean()+1e-12)
                target=gradient_norms.detach().mean()*(inverse**float(args.gradnorm_alpha))
                weight_optimizer.zero_grad(set_to_none=True); torch.abs(gradient_norms-target).sum().backward(retain_graph=True); weight_optimizer.step()
                with torch.no_grad():
                    task_weights.clamp_(min=1e-3); task_weights.mul_(2.0/(task_weights.sum()+1e-12)); coefficient_rows.append(float(task_weights[1].detach().cpu()))
                optimizer.zero_grad(set_to_none=True); total=(task_weights.detach()*torch.stack(tasks)).sum(); total.backward()
            else: raise ValueError('unknown baseline '+variant)
            norm_before=torch.nn.utils.clip_grad_norm_(parameters,float(args.gradient_clip)); clipping_count+=int(float(norm_before)>float(args.gradient_clip)); optimizer.step(); losses_epoch.append(float(total.detach().cpu()))
        validation_logits=_predict(model,roles['validation']['signals'],device,args.eval_batch_size)
        probability=1.0/(1.0+np.exp(-np.clip(validation_logits[:,COMMON_INDICES],-40.0,40.0)))
        targets=roles['validation']['targets6'][:,COMMON_INDICES]; weak=weak6[roles['validation']['sample_ids']][:,COMMON_INDICES]
        macro=_macro_aupr(targets,probability); rank=_target_weak_rank(targets,probability,weak); key=(macro,-float('inf') if not np.isfinite(rank) else rank)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses_epoch)),"validation_common_macro_aupr":macro,"validation_target_weak_auroc":rank})
        if key>(best_key[0]+1e-8,best_key[1]) or (np.isclose(key[0],best_key[0]) and key[1]>best_key[1]+1e-8):
            best_key=key; best_epoch=epoch; best_state=copy.deepcopy({name:value.detach().cpu() for name,value in model.state_dict().items()}); stale=0
        else:
            stale+=1
            if stale>=int(args.patience): break
    if best_state is None: raise RuntimeError('baseline training produced no checkpoint')
    model.load_state_dict(best_state); output_dir.mkdir(parents=True,exist_ok=True)
    diagnostics={"mean_task_cosine":float(np.mean(cosine_rows)) if cosine_rows else None,"conflict_batch_fraction":float(conflict_count/len(cosine_rows)) if cosine_rows else None,"coefficient_mean":float(np.mean(coefficient_rows)) if coefficient_rows else None,"coefficient_last":float(coefficient_rows[-1]) if coefficient_rows else None,"gradient_clipping_count":clipping_count}
    torch.save({"model_state_dict":best_state,"variant":variant,"best_epoch":best_epoch,"train_seed":train_seed,"diagnostics":diagnostics},output_dir/'best_model.pth'); _write_history(output_dir/'history.csv',history)
    return model,{"best_epoch":best_epoch,"epochs_ran":len(history),"best_common_macro_aupr":float(best_key[0]),"best_target_weak_auroc":float(best_key[1]),"train_seconds":float(time.time()-started),"gradient_diagnostics":diagnostics,"trainable_shared_parameters":sum(p.numel() for p in scope['shared']),"trainable_classifier_parameters":sum(p.numel() for p in scope['classifier'])}

def _run_seed(args,config,labels6,weak6,seed,device,data_sha256):
    seed_dir=Path(args.output_root)/f'seed_{seed}'; summary_path=seed_dir/'seed_summary.json'
    if args.resume and summary_path.is_file(): return _read_json(summary_path)
    split,manifest,ids,roles=_build_roles(args,config,labels6,seed); f2_dir=Path(args.f2_root)/'runs'/F2_MODEL_NAME/f'seed_{seed}'
    template,checkpoint_path=_load_backbone(f2_dir,F2_MODEL_NAME,device); initial=copy.deepcopy({k:v.detach().cpu() for k,v in template.state_dict().items()}); del template
    states={}; training={}; variants=[]
    for variant,masked in [('C6_Joint',None),('M5_GradientIsolated',1)]:
        model=_load_model_from_state(f2_dir,device,initial); model,training[variant]=_train_endpoint(model,variant,masked,roles,weak6,device,seed,args,seed_dir/variant); states[variant]=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}); row=_evaluate_model_v4(variant,model,roles,weak6,seed_dir,device,args); row['status']='valid'; variants.append(row); del model
    for variant in TRAINED_BASELINES:
        model=_load_model_from_state(f2_dir,device,initial); model,training[variant]=_train_baseline(model,variant,roles,weak6,device,seed,args,seed_dir/variant); row=_evaluate_model_v4(variant,model,roles,weak6,seed_dir,device,args); row['status']='valid'; variants.append(row); del model
    direction=_state_direction(states['C6_Joint'],states['M5_GradientIsolated'])
    variants.append(_save_and_evaluate('ModelAverage0p5',_apply_direction(states['C6_Joint'],direction,0.5),{"source":"paired_midpoint","scale":0.5,"direction_norm":_direction_norm(direction)},f2_dir,roles,weak6,seed_dir,device,args))
    variants.append(_save_and_evaluate('FullAlpha1p5',_apply_direction(states['C6_Joint'],direction,1.5),{"source":"frozen_paired_task_vector","scale":1.5,"direction_norm":_direction_norm(direction)},f2_dir,roles,weak6,seed_dir,device,args))
    _write_json(seed_dir/'longtail_manifest.json',manifest.to_dict()); np.savez_compressed(str(seed_dir/'role_indices.npz'),manifest=manifest.selected_indices,**ids)
    payload={"seed":seed,"status":"completed","smoke":bool(args.smoke),"data_sha256":data_sha256,"split_sha256":split.sha256,"manifest_sha256":manifest.sha256,"role_indices_sha256":_indices_sha256(*[ids[r] for r in ('fit','validation','early','calibration','threshold')]),"f2_checkpoint":str(checkpoint_path),"f2_checkpoint_sha256":_sha256(checkpoint_path),"training":training,"variants":variants,"role_counts":{k:len(v) for k,v in ids.items()},"identity_audit_passed":True,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"test_v4_used":False,"test_v5_used":False,"test_v6_used":False}
    _write_json(summary_path,payload); return payload

def _aggregate_and_comparisons(seed_rows):
    paths={"macro_aupr":('metrics','macro_aupr'),"macro_f1":('metrics','macro_f1'),"target_mean_weak_recall":('metrics','target_mean_weak_recall'),"target_mean_negative_fpr":('metrics','target_mean_negative_fpr'),"sawfm_present_macro_f1":('metrics','subgroups','sawfm_present','macro_f1'),"sawfm_absent_macro_f1":('metrics','subgroups','sawfm_absent','macro_f1')}
    aggregate={}; comparisons={}
    for vi,variant in enumerate(VARIANTS):
        rows=[_variant_map(s)[variant] for s in seed_rows]; aggregate[variant]={name:_mean_ci([_metric(row,path) for row in rows],20262600+vi*100+i) for i,(name,path) in enumerate(paths.items())}
        aggregate[variant]['per_label_aupr']={name:_mean_ci([row['metrics']['per_label_diagnostics'][name]['aupr'] for row in rows],20262700+vi*100+i) for i,name in enumerate(COMMON_NAMES)}
        aggregate[variant]['per_label_weak_recall']={name:_mean_ci([row['metrics']['per_label_diagnostics'][name]['weak_recall'] for row in rows],20262800+vi*100+i) for i,name in enumerate(TARGET_NAMES)}
        aggregate[variant]['valid_seed_count']=len(rows)
        if variant=='C6_Joint': continue
        per=[]
        for seed_row,row in zip(seed_rows,rows):
            base=_variant_map(seed_row)['C6_Joint']; delta={name:_metric(row,path)-_metric(base,path) for name,path in paths.items()}; delta['per_label_aupr']={name:row['metrics']['per_label_diagnostics'][name]['aupr']-base['metrics']['per_label_diagnostics'][name]['aupr'] for name in COMMON_NAMES}; delta['per_label_weak_recall']={name:row['metrics']['per_label_diagnostics'][name]['weak_recall']-base['metrics']['per_label_diagnostics'][name]['weak_recall'] for name in TARGET_NAMES}; per.append({'seed':seed_row['seed'],'delta':delta})
        agg={name:_mean_ci([r['delta'][name] for r in per],20262900+i) for i,name in enumerate(paths)}; agg['per_label_aupr']={name:_mean_ci([r['delta']['per_label_aupr'][name] for r in per],20263000+i) for i,name in enumerate(COMMON_NAMES)}; agg['per_label_weak_recall']={name:_mean_ci([r['delta']['per_label_weak_recall'][name] for r in per],20263100+i) for i,name in enumerate(TARGET_NAMES)}; agg['positive_target_weak_recall_seeds']=sum(r['delta']['target_mean_weak_recall']>0 for r in per)
        comparisons[variant]={"candidate":variant,"baseline":"C6_Joint","per_seed":per,"aggregate":agg,"valid_seed_count":len(per)}
    return aggregate,comparisons

def _candidate_gate(comparison,absolute):
    d=comparison['aggregate']; checks={"valid_5_of_5":comparison['valid_seed_count']==5,"macro_aupr_noninferior":d['macro_aupr']['mean']>=-0.005,"macro_f1_gain":d['macro_f1']['mean']>=0.010,"target_weak_gain":d['target_mean_weak_recall']['mean']>=0.025,"each_target_weak_nonnegative":all(d['per_label_weak_recall'][n]['mean']>=0 for n in TARGET_NAMES),"target_fpr_cap":absolute['target_mean_negative_fpr']['mean']<=0.20,"positive_weak_4_of_5":d['positive_target_weak_recall_seeds']>=4,"saw_present_positive":d['sawfm_present_macro_f1']['mean']>0,"saw_absent_noninferior":d['sawfm_absent_macro_f1']['mean']>=-0.005,"target_aupr_noninferior":all(d['per_label_aupr'][n]['mean']>=-0.010 for n in TARGET_NAMES)}; return {"checks":checks,"passed":all(checks.values())}

def _strongest_baseline(comparisons,aggregate):
    rows=[]; audit={}
    for name in VARIANTS:
        if name in ('C6_Joint','FullAlpha1p5'): continue
        d=comparisons[name]['aggregate']; checks={"valid_5_of_5":comparisons[name]['valid_seed_count']==5,"macro_aupr_noninferior":d['macro_aupr']['mean']>=-0.005,"target_fpr_cap":aggregate[name]['target_mean_negative_fpr']['mean']<=0.20,"saw_absent_noninferior":d['sawfm_absent_macro_f1']['mean']>=-0.010}; passed=all(checks.values()); audit[name]={"checks":checks,"eligible":passed}
        if passed: rows.append((name,min(d['per_label_weak_recall'][n]['mean'] for n in TARGET_NAMES),d['target_mean_weak_recall']['mean'],d['macro_f1']['mean']))
    rows.sort(key=lambda x:(-x[1],-x[2],-x[3],x[0])); return (rows[0][0] if rows else None),audit,rows

def _paired_candidate_vs_baseline(seed_rows,baseline):
    paths={"macro_aupr":('metrics','macro_aupr'),"macro_f1":('metrics','macro_f1'),"target_mean_weak_recall":('metrics','target_mean_weak_recall'),"target_mean_negative_fpr":('metrics','target_mean_negative_fpr'),"sawfm_present_macro_f1":('metrics','subgroups','sawfm_present','macro_f1'),"sawfm_absent_macro_f1":('metrics','subgroups','sawfm_absent','macro_f1')}; per=[]
    for seed_row in seed_rows:
        v=_variant_map(seed_row); per.append({"seed":seed_row['seed'],"delta":{name:_metric(v['FullAlpha1p5'],path)-_metric(v[baseline],path) for name,path in paths.items()}})
    return {"per_seed":per,"aggregate":{name:_mean_ci([r['delta'][name] for r in per],20263200+i) for i,name in enumerate(paths)}}

def _advantage_gate(candidate_vs_baseline,baseline,comparisons,aggregate):
    d=candidate_vs_baseline['aggregate']; macro_adv=d['macro_f1']['mean']>=0.005 and d['macro_f1']['bootstrap_95_ci'][0]>0; weak_adv=d['target_mean_weak_recall']['mean']>=0.005 and d['target_mean_weak_recall']['bootstrap_95_ci'][0]>0
    checks={"strongest_baseline_exists":baseline is not None,"macro_aupr_noninferior":d['macro_aupr']['mean']>=-0.003,"macro_f1_not_worse":d['macro_f1']['mean']>=-0.003,"weak_recall_not_worse":d['target_mean_weak_recall']['mean']>=-0.003,"one_primary_advantage_0p005_with_positive_ci":macro_adv or weak_adv,"target_fpr_delta_le_0p01":d['target_mean_negative_fpr']['mean']<=0.01,"saw_present_noninferior":d['sawfm_present_macro_f1']['mean']>=-0.005,"saw_absent_noninferior":d['sawfm_absent_macro_f1']['mean']>=-0.005}; return {"strongest_baseline":baseline,"checks":checks,"passed":all(checks.values()),"macro_advantage":macro_adv,"weak_advantage":weak_adv}

def run(args):
    config=load_config(args.config); _validate_v6_config(config); identity=_read_json(args.identity_audit)
    if not identity.get('identity_audit_passed') or identity.get('test_files_opened'): raise ValueError('development-v6 identity audit failed')
    if any(identity.get(name,True) for name in ('locked_test_used','test_v2_used','test_v3_used','test_v4_used','test_v5_used','test_v6_used')): raise ValueError('forbidden role use in identity audit')
    data_path=_data_path(config); data_sha256=_sha256(data_path)
    if identity.get('sha256','').lower()!=data_sha256.lower(): raise ValueError('development-v6 hash mismatch')
    labels6=read_labels(data_path); parameters,_=read_parameters(data_path); weak6=compute_weak_component_mask(labels6,_component_power(parameters),float(args.weak_threshold_db)); device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu')); args.epochs=int(args.smoke_epochs) if args.smoke else int(args.epochs)
    seed_rows=[_run_seed(args,config,labels6,weak6,int(seed),device,data_sha256) for seed in args.seeds]; aggregate,comparisons=_aggregate_and_comparisons(seed_rows)
    if args.smoke: candidate_gate={"passed":False,"checks":{},"reason":"smoke_only"}; strongest=None; baseline_audit={}; ranking=[]; paired={}; advantage={"passed":False,"checks":{},"reason":"smoke_only"}; decision='smoke_only_no_method_claim'
    else:
        candidate_gate=_candidate_gate(comparisons['FullAlpha1p5'],aggregate['FullAlpha1p5']); strongest,baseline_audit,ranking=_strongest_baseline(comparisons,aggregate); paired=_paired_candidate_vs_baseline(seed_rows,strongest) if strongest else {}; advantage=_advantage_gate(paired,strongest,comparisons,aggregate) if strongest else {"passed":False,"checks":{"strongest_baseline_exists":False}}
        decision='promote_to_robustness_stage' if candidate_gate['passed'] and advantage['passed'] else ('downgrade_to_protocol_specific_task_vector' if candidate_gate['passed'] else 'stop_innovation_route')
    payload={"status":"completed","experiment":"route_a_development_v6_multitask_baselines","evidence_level":"smoke" if args.smoke else "development","smoke":bool(args.smoke),"device":str(device),"protocol":{"seeds":[int(s) for s in args.seeds],"variants":list(VARIANTS),"epochs":int(args.epochs),"task_definition":{"common":"common-five BCE mean","sawfm":"0.2 * SawFM BCE"},"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"test_v4_used":False,"test_v5_used":False,"test_v6_used":False},"data":{"path":str(data_path),"sha256":data_sha256,"sample_count":len(labels6),"identity_audit":identity},"per_seed":seed_rows,"aggregate":aggregate,"comparisons_vs_c6":comparisons,"candidate_effect_gate":candidate_gate,"baseline_eligibility":baseline_audit,"baseline_ranking":ranking,"strongest_baseline":strongest,"candidate_vs_strongest":paired,"method_advantage_gate":advantage,"next_decision":decision,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"test_v4_used":False,"test_v5_used":False,"test_v6_used":False}
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True); _write_json(root/'route_a_development_v6_multitask_baselines_summary.json',payload); _write_json(root/'run_receipt.json',{"status":"completed","evidence_level":payload['evidence_level'],"data_sha256":data_sha256,"identity_audit_passed":True,"candidate_effect_gate_passed":candidate_gate['passed'],"strongest_baseline":strongest,"method_advantage_gate_passed":advantage['passed'],"next_decision":decision,"locked_test_used":False,"test_v2_used":False,"test_v3_used":False,"test_v4_used":False,"test_v5_used":False,"test_v6_used":False,"test_access_allowed":False}); print(json.dumps({"status":"completed","evidence_level":payload['evidence_level'],"candidate_effect_gate":candidate_gate,"strongest_baseline":strongest,"method_advantage_gate":advantage,"next_decision":decision},ensure_ascii=False,indent=2)); return payload

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs'/'student_a'/'six_class_development_v6.yaml'); p.add_argument('--identity-audit',type=Path,default=PROJECT_ROOT/'outputs'/'tapc'/'route_a_development_v6_generation'/'identity_audit.json'); p.add_argument('--f2-root',type=Path,default=PROJECT_ROOT/'outputs'/'tapc'/'route_a_dfms_iq_structure_screen'); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs'/'tapc'/'route_a_development_v6_multitask_baselines'); p.add_argument('--seeds',nargs='+',type=int,default=[2039,2040,2041,2042,2043]); p.add_argument('--epochs',type=int,default=12); p.add_argument('--patience',type=int,default=4); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--io-batch-size',type=int,default=128); p.add_argument('--eval-batch-size',type=int,default=128); p.add_argument('--learning-rate',type=float,default=5e-4); p.add_argument('--representation-lr-scale',type=float,default=0.1); p.add_argument('--saw-loss-weight',type=float,default=0.2); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--gradient-clip',type=float,default=5.0); p.add_argument('--max-negative-fpr',type=float,default=0.20); p.add_argument('--weak-threshold-db',type=float,default=10.0); p.add_argument('--cagrad-c',type=float,default=0.4); p.add_argument('--cagrad-grid',type=int,default=101); p.add_argument('--gradnorm-alpha',type=float,default=1.5); p.add_argument('--gradnorm-weight-lr',type=float,default=0.025); p.add_argument('--device',default=None); p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--smoke-epochs',type=int,default=2); p.add_argument('--smoke-train-samples',type=int,default=128); p.add_argument('--smoke-eval-samples',type=int,default=64); return p.parse_args()
if __name__=='__main__': run(parse_args())

