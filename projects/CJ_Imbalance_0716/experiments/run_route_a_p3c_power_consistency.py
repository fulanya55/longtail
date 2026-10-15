"""P3c: paired-power consistency regularization on the six-class Route-A data."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from experiments.run_route_a_c1_baseline import CLASS_NAMES, SELECTED_INDICES, NineClassDataset, metrics, pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB, find_file, profile_cost, receiver_seeds, sha256, write_json
from experiments.run_route_a_p3_dual_axis_longtail import load_shared_f2, set_seed

VARIANTS=("F2_BCE_30dB","F2_BCE_power_consistency"); SEEDS=(2039,2040,2041)

class PairedPowerDataset(Dataset):
    def __init__(self, paths, indices):
        self.datasets=[NineClassDataset(p,indices,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)) for p in paths]
    def __len__(self): return len(self.datasets[0])
    def __getitem__(self,i):
        a,b,c=(d[int(i)] for d in self.datasets)
        return {"signal_30":a["signal"],"signal_low":b["signal"],"signal_high":c["signal"],"label":a["label"]}
    def close(self):
        for d in self.datasets: d.close()

def predict(model,path,idx,device,batch):
    ds=NineClassDataset(path,idx,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64)); dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True); y=[]; l=[]
    try:
        model.eval()
        with torch.no_grad():
            for b in dl: l.append(model(b['signal'].to(device,non_blocking=True)).cpu().numpy()); y.append(b['label'].numpy())
    finally: ds.close()
    return np.concatenate(y).astype(np.uint8),np.concatenate(l).astype(np.float32)

def run_one(args,name,seed,paths,fit,val,power_dev,out):
    set_seed(seed); model=__import__('cjlt.models.backbone_adapter',fromlist=['build_backbone']).build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(args.device); init=load_shared_f2(model,args.f2_checkpoint)
    ds=PairedPowerDataset(paths,fit); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); hist=[]; best=-1.; best_ep=0; started=time.time()
    params=[p for p in model.parameters() if p.requires_grad]
    grad_cosines=[]; grad_conflicts=[]

    def _zerosafe(g,p):
        return torch.zeros_like(p) if g is None else g

    def _dot(gs1,gs2):
        return sum(torch.sum(_zerosafe(a,p)*_zerosafe(b,p)) for p,a,b in zip(params,gs1,gs2))

    def _sqnorm(gs):
        return sum(torch.sum(_zerosafe(g,p)**2) for p,g in zip(params,gs))

    try:
        for ep in range(1,args.epochs+1):
            model.train(); losses=[]; conses=[]; ep_cos=[]; ep_conf=[]
            for b in dl:
                x=b['signal_30'].to(args.device,non_blocking=True); xl=b['signal_low'].to(args.device,non_blocking=True); xh=b['signal_high'].to(args.device,non_blocking=True); y=b['label'].to(args.device,non_blocking=True); opt.zero_grad(set_to_none=True); z=model(x); zl=model(xl); zh=model(xh); lc=nn.functional.binary_cross_entropy_with_logits(z,y)
                if name=='F2_BCE_power_consistency':
                    low_cons=nn.functional.mse_loss(torch.sigmoid(zl),torch.sigmoid(z).detach())
                    high_cons=nn.functional.mse_loss(torch.sigmoid(zh),torch.sigmoid(z).detach())
                    if args.consistency_mode=='high_only': consistency=high_cons; loss=lc+args.consistency_weight*consistency
                    elif args.consistency_mode=='low_only': consistency=low_cons; loss=lc+args.consistency_weight*consistency
                    elif args.consistency_mode=='pcgrad':
                        # PCGrad projects the low/high consistency gradients when they conflict.
                        g_cls=torch.autograd.grad(lc,params,retain_graph=True,allow_unused=True)
                        g_low=torch.autograd.grad(low_cons,params,retain_graph=True,allow_unused=True)
                        g_high=torch.autograd.grad(high_cons,params,retain_graph=True,allow_unused=True)
                        n_low=_sqnorm(g_low); n_high=_sqnorm(g_high); dot_lh=_dot(g_low,g_high)
                        denom=(torch.sqrt(n_low*n_high)+1e-12)
                        cosine=dot_lh/denom
                        conflict=bool((dot_lh<0).detach().cpu())
                        ep_cos.append(float(cosine.detach().cpu())); ep_conf.append(float(conflict))
                        if conflict and float(n_high.detach().cpu())>0:
                            g_low=[_zerosafe(g,p)-dot_lh/n_high*_zerosafe(h,p) for p,g,h in zip(params,g_low,g_high)]
                        if conflict and float(n_low.detach().cpu())>0:
                            g_high=[_zerosafe(g,p)-dot_lh/n_low*_zerosafe(l,p) for p,g,l in zip(params,g_high,g_low)]
                        for p,gc,gl,gh in zip(params,g_cls,g_low,g_high):
                            p.grad=_zerosafe(gc,p)+args.consistency_weight*(_zerosafe(gl,p)+_zerosafe(gh,p))/2.0
                        consistency=(low_cons+high_cons)/2.0; loss=lc+args.consistency_weight*consistency
                    else: consistency=(low_cons+high_cons)/2.0; loss=lc+args.consistency_weight*consistency
                else: consistency=z.sum()*0; loss=lc
                if not (name=='F2_BCE_power_consistency' and args.consistency_mode=='pcgrad'):
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params,5.0); opt.step(); losses.append(float(loss.detach().cpu())); conses.append(float(consistency.detach().cpu()))
            yt,yl=predict(model,paths[0],val,args.device,args.eval_batch_size); vm=metrics(yt,yl); hist.append({'epoch':ep,'train_loss':float(np.mean(losses)),'consistency_loss':float(np.mean(conses)),'gradient_cosine_mean':float(np.mean(ep_cos)) if ep_cos else None,'gradient_conflict_fraction':float(np.mean(ep_conf)) if ep_conf else None,'validation':vm})
            if vm['macro_aupr']>best: best=vm['macro_aupr']; best_ep=ep; torch.save(model.state_dict(),out/f'{name}__seed_{seed}.pth')
    finally: ds.close()
    model.load_state_dict(torch.load(str(out/f'{name}__seed_{seed}.pth'),map_location=args.device,weights_only=True)); r={'variant':name,'seed':seed,'history':hist,'best_epoch':best_ep,'validation':hist[best_ep-1]['validation'],'power_dev':{},'total_parameters':int(sum(p.numel() for p in model.parameters())),'cost_profile':profile_cost(model,args.device),'initialization':init,'consistency_weight':args.consistency_weight,'consistency_mode':args.consistency_mode,'locked_test_used':False,'elapsed_seconds':time.time()-started}
    for p,(pp,ii) in power_dev.items(): yt,yl=predict(model,pp,ii,args.device,args.eval_batch_size); r['power_dev'][str(p)]=metrics(yt,yl)
    r['checkpoint_sha256']=sha256(out/f'{name}__seed_{seed}.pth'); write_json(out/f'{name}__seed_{seed}.json',r); print(json.dumps({'variant':name,'seed':seed,'val_macro_aupr':r['validation']['macro_aupr'],'val_combo':r['validation']['combination_exact_accuracy']},ensure_ascii=False),flush=True); return r

def run(args):
    dev=torch.device(args.device)
    if dev.type!='cuda': raise RuntimeError('P3c requires CUDA')
    args.device=dev; args.output_root.mkdir(parents=True,exist_ok=True); paths={p:find_file(args.data_root,p,'train') for p in [10,30,50]}; base=paths[30]; pure=pure_six_indices(base); perm=np.random.default_rng(args.split_seed).permutation(pure); fit=np.sort(perm[:args.fit_samples]); val=np.sort(perm[args.fit_samples:args.fit_samples+args.val_samples]); used=set(receiver_seeds(base,np.sort(np.concatenate([fit,val])))); power_dev={}
    for p in POWERS_DB:
        pp=find_file(args.data_root,p,'train'); pi=pure_six_indices(pp); rs=np.asarray(receiver_seeds(pp,pi),dtype=np.int64); cand=pi[~np.isin(rs,np.fromiter(used,dtype=np.int64))]; ii=np.sort(np.random.default_rng(args.split_seed+p).choice(cand,size=min(args.power_dev_samples,len(cand)),replace=False)); power_dev[p]=(pp,ii)
    role={'schema':'route-a-p3c-power-consistency-role-manifest-v1','selected_classes':list(CLASS_NAMES),'paired_power_views':[10,30,50],'pairing_key':'receiver_noise_seed and identical sample order across train views','consistency_mode':args.consistency_mode,'fit':{'path':str(base),'power_db':30,'count':int(len(fit)),'sample_indices_sha256':__import__('experiments.run_route_a_p3_dual_axis_longtail',fromlist=['sample_id_sha256']).sample_id_sha256(fit),'receiver_noise_seed':sorted(set(receiver_seeds(base,fit)))},'validation':{'path':str(base),'power_db':30,'count':int(len(val)),'sample_indices_sha256':__import__('experiments.run_route_a_p3_dual_axis_longtail',fromlist=['sample_id_sha256']).sample_id_sha256(val),'receiver_noise_seed':sorted(set(receiver_seeds(base,val)))},'power_development':{'source_role':'train_only_unseen_identity','count_by_power':{str(p):len(i) for p,(pp,i) in power_dev.items()}},'test':{'used':False,'locked_test_used':False}}; write_json(args.output_root/'role_manifest.json',role)
    rows=[]
    for seed in args.seeds:
        for name in VARIANTS:
            out=args.output_root/name; out.mkdir(parents=True,exist_ok=True); rows.append(run_one(args,name,int(seed),[paths[30],paths[10],paths[50]],fit,val,power_dev,out))
    summary={'schema':'route-a-p3c-power-consistency-summary-v1','status':'completed','locked_test_used':False,'protocol':{'paired_powers':[10,30,50],'epochs':args.epochs,'consistency_weight':args.consistency_weight,'consistency_mode':args.consistency_mode,'seeds':list(args.seeds)},'variants':{}}
    for name in VARIANTS:
        rr=[r for r in rows if r['variant']==name]; summary['variants'][name]={'parameters':rr[0]['total_parameters'],'val_macro_aupr_mean':float(np.mean([r['validation']['macro_aupr'] for r in rr])),'val_macro_aupr_std':float(np.std([r['validation']['macro_aupr'] for r in rr],ddof=1)),'val_macro_f1_mean':float(np.mean([r['validation']['macro_f1'] for r in rr])),'val_combo_mean':float(np.mean([r['validation']['combination_exact_accuracy'] for r in rr])),'power_dev_macro_aupr_mean_by_power':{str(p):float(np.mean([r['power_dev'][str(p)]['macro_aupr'] for r in rr])) for p in POWERS_DB},'power_dev_combo_mean_by_power':{str(p):float(np.mean([r['power_dev'][str(p)]['combination_exact_accuracy'] for r in rr])) for p in POWERS_DB}}
    write_json(args.output_root/'p3c_power_consistency_summary.json',summary); write_json(args.output_root/'terminal_receipt.json',{'status':'completed','summary':str((args.output_root/'p3c_power_consistency_summary.json').resolve()),'locked_test_used':False,'test_files_opened':False,'script_sha256':sha256(Path(__file__))}); print(json.dumps({'status':'completed','summary':str(args.output_root/'p3c_power_consistency_summary.json')},ensure_ascii=False),flush=True)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path(r'E:\论文相关\干扰数据\复合干扰数据0716\九类')); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p3c_power_consistency_20260829'); p.add_argument('--f2-checkpoint',type=Path,default=PROJECT_ROOT/'outputs/route_a_c1_baseline_full_20260826/best_model.pth'); p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS)); p.add_argument('--split-seed',type=int,default=260830); p.add_argument('--fit-samples',type=int,default=4096); p.add_argument('--val-samples',type=int,default=2048); p.add_argument('--power-dev-samples',type=int,default=256); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--eval-batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=2); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--consistency-weight',type=float,default=.1); p.add_argument('--consistency-mode',choices=('both','high_only','low_only','pcgrad'),default='both'); p.add_argument('--device',default='cuda:0'); return p.parse_args()
if __name__=='__main__': run(parse_args())
