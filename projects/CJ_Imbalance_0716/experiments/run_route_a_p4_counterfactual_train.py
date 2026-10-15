"""P4 counterfactual preflight training on generator-derived paired samples."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import CLASS_NAMES, metrics
from experiments.run_route_a_p4_matched_hard_negative import load_shared
from experiments.run_route_a_nineclass_feature_sufficiency import profile_cost, sha256, write_json

SEEDS=(2039,2040,2041); VARIANTS=('F2_BCE','F2_BCE_counterfactualHN')

def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def norm(x):
    x=np.asarray(x,dtype=np.float32); rms=np.sqrt(np.mean(x[0]**2+x[1]**2)); return x/np.sqrt(rms*rms+1e-12) if np.isfinite(rms) and rms>0 else x

class PairDataset(Dataset):
    def __init__(self,iq_a,iq_b,la,lb): self.a=iq_a; self.b=iq_b; self.la=la; self.lb=lb
    def __len__(self): return len(self.a)
    def __getitem__(self,i): return {'a':torch.from_numpy(norm(np.stack([self.a[i].real,self.a[i].imag]))),'b':torch.from_numpy(norm(np.stack([self.b[i].real,self.b[i].imag]))),'la':torch.from_numpy(self.la[i].astype(np.float32)),'lb':torch.from_numpy(self.lb[i].astype(np.float32))}

def read_pairs(path):
    with h5py.File(str(path),'r') as h:
        # MATLAB column-major arrays are exposed by h5py with the sample
        # dimension first for iq_a/iq_b; compound real/imag fields are decoded.
        ra=np.asarray(h['iq_a'][:]); rb=np.asarray(h['iq_b'][:]);
        ia=ra['real'].astype(np.float64)+1j*ra['imag'].astype(np.float64) if ra.dtype.fields else ra
        ib=rb['real'].astype(np.float64)+1j*rb['imag'].astype(np.float64) if rb.dtype.fields else rb
        la=np.asarray(h['labels_a'][:]).T.astype(np.uint8); lb=np.asarray(h['labels_b'][:]).T.astype(np.uint8); group=np.asarray(h['pair_group_id'][:]).reshape(-1).astype(np.int64)
    return ia,ib,la,lb,group

def eval_pairs(model,ds,device,batch):
    dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0); ys=[];ls=[]
    model.eval()
    with torch.no_grad():
        for b in dl:
            for key in ('a','b'):
                ls.append(model(b[key].to(device)).cpu().numpy())
            ys.extend([b['la'].numpy(),b['lb'].numpy()])
    return metrics(np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32))

def run_one(args,name,seed,fit_ds,val_ds,out):
    set_seed(seed); model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=.2).to(args.device); init=load_shared(model,args.f2_checkpoint); dl=DataLoader(fit_ds,batch_size=args.batch_size,shuffle=True,num_workers=0); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); hist=[];best=-1.;best_ep=0;started=time.time()
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]; hns=[]
        for b in dl:
            a=b['a'].to(args.device); bb=b['b'].to(args.device); la=b['la'].to(args.device); lb=b['lb'].to(args.device); opt.zero_grad(set_to_none=True); za=model(a); zb=model(bb); lc=(nn.functional.binary_cross_entropy_with_logits(za,la)+nn.functional.binary_cross_entropy_with_logits(zb,lb))/2
            diff=la-lb; mask=diff.abs()>0
            if name=='F2_BCE_counterfactualHN' and bool(mask.any()):
                sign=torch.where(diff>0,torch.ones_like(diff),-torch.ones_like(diff)); lh=torch.relu(args.margin-sign*(za-zb))[mask].mean(); loss=lc+args.hn_weight*lh
            else: lh=za.sum()*0; loss=lc
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu())); hns.append(float(lh.detach().cpu()))
        vm=eval_pairs(model,val_ds,args.device,args.eval_batch_size); hist.append({'epoch':ep,'train_loss':float(np.mean(losses)),'counterfactual_hn_loss':float(np.mean(hns)),'validation':vm})
        if vm['macro_aupr']>best: best=vm['macro_aupr']; best_ep=ep; torch.save(model.state_dict(),out/f'{name}__seed_{seed}.pth')
    model.load_state_dict(torch.load(str(out/f'{name}__seed_{seed}.pth'),map_location=args.device,weights_only=True)); r={'variant':name,'seed':seed,'history':hist,'best_epoch':best_ep,'validation':hist[best_ep-1]['validation'],'total_parameters':int(sum(p.numel() for p in model.parameters())),'cost_profile':profile_cost(model,args.device),'initialization':init,'hn_weight':args.hn_weight,'margin':args.margin,'locked_test_used':False,'elapsed_seconds':time.time()-started,'checkpoint_sha256':sha256(out/f'{name}__seed_{seed}.pth')}; write_json(out/f'{name}__seed_{seed}.json',r); print(json.dumps({'variant':name,'seed':seed,'val_combo':r['validation']['combination_exact_accuracy']},ensure_ascii=False),flush=True); return r

def run(args):
    device=torch.device(args.device)
    if device.type!='cuda': raise RuntimeError('counterfactual preflight requires CUDA')
    args.device=device; args.output_root.mkdir(parents=True,exist_ok=True); ia,ib,la,lb,g=read_pairs(args.data_file); fit_mask=(g>=1) & (g<=args.fit_group_count); val_mask=(g>args.fit_group_count) & (g<=args.pair_groups)
    fit_ds=PairDataset(ia[fit_mask],ib[fit_mask],la[fit_mask],lb[fit_mask]); val_ds=PairDataset(ia[val_mask],ib[val_mask],la[val_mask],lb[val_mask]); role={'schema':'route-a-p4-counterfactual-preflight-role-manifest-v1','source_file':str(args.data_file.resolve()),'pair_groups':args.pair_groups,'fit_group_count':args.fit_group_count,'validation_group_count':args.pair_groups-args.fit_group_count,'samples_per_pair':args.samples_per_pair,'fit_pair_count':int(fit_mask.sum()),'validation_pair_count':int(val_mask.sum()),'fit_groups':list(range(1,args.fit_group_count+1)),'validation_groups':list(range(args.fit_group_count+1,args.pair_groups+1)),'split_rule':'entire pair groups held out for validation; no combination-pair group overlap','same_sample_seed_within_pair':True,'locked_test_used':False}; write_json(args.output_root/'role_manifest.json',role); rows=[]
    for seed in args.seeds:
        for name in VARIANTS:
            out=args.output_root/name; out.mkdir(parents=True,exist_ok=True); rows.append(run_one(args,name,int(seed),fit_ds,val_ds,out))
    summary={'schema':'route-a-p4-counterfactual-preflight-summary-v1','status':'completed','locked_test_used':False,'protocol':{'seeds':list(args.seeds),'epochs':args.epochs,'hn_weight':args.hn_weight,'margin':args.margin},'variants':{}}
    for name in VARIANTS:
        rr=[r for r in rows if r['variant']==name]; summary['variants'][name]={'parameters':rr[0]['total_parameters'],'val_macro_aupr_mean':float(np.mean([r['validation']['macro_aupr'] for r in rr])),'val_macro_f1_mean':float(np.mean([r['validation']['macro_f1'] for r in rr])),'val_combo_mean':float(np.mean([r['validation']['combination_exact_accuracy'] for r in rr]))}
    write_json(args.output_root/'p4_counterfactual_preflight_summary.json',summary); write_json(args.output_root/'terminal_receipt.json',{'status':'completed','summary':str((args.output_root/'p4_counterfactual_preflight_summary.json').resolve()),'locked_test_used':False,'test_files_opened':False,'script_sha256':sha256(Path(__file__))}); print(json.dumps({'status':'completed','summary':str(args.output_root/'p4_counterfactual_preflight_summary.json')},ensure_ascii=False))

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-file',type=Path,default=PROJECT_ROOT/'outputs/route_a_p4_counterfactual_train_dev_20260901/route_a_p4_counterfactual_preflight.mat'); p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p4_counterfactual_train_20260901'); p.add_argument('--f2-checkpoint',type=Path,default=PROJECT_ROOT/'outputs/route_a_c1_baseline_full_20260826/best_model.pth'); p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS)); p.add_argument('--pair-groups',type=int,default=6); p.add_argument('--fit-group-count',type=int,default=3); p.add_argument('--samples-per-pair',type=int,default=64); p.add_argument('--fit-repeats',type=int,default=48); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--eval-batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=4); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--hn-weight',type=float,default=.1); p.add_argument('--margin',type=float,default=.25); p.add_argument('--device',default='cuda:0'); return p.parse_args()
if __name__=='__main__': run(parse_args())
