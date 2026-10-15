"""P2c: F2 baseline versus a bounded visibility auxiliary head."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_c1_baseline import CLASS_NAMES,SELECTED_INDICES,NineClassDataset,metrics,pure_six_indices
from experiments.run_route_a_nineclass_feature_sufficiency import POWERS_DB,find_file,receiver_seeds,profile_cost,sha256,write_json

def visibility_targets(path,indices):
    indices=np.asarray(indices,dtype=np.int64)
    with h5py.File(str(path),'r') as h:
        labels=np.asarray(h['known_label'][list(SELECTED_INDICES),:][:,indices],dtype=np.float32).T
        refs=h['weak_component_mask'][0,indices]
        target=np.zeros_like(labels,dtype=np.float32); mask=np.zeros_like(labels,dtype=np.float32)
        for row,ref in enumerate(refs):
            weak=np.asarray(h[ref][:],dtype=np.float32).reshape(-1)
            active=np.flatnonzero(labels[row]>0)
            if len(weak)!=len(active): raise RuntimeError(f'weak mask alignment mismatch at {indices[row]}: {len(weak)} vs {len(active)}')
            target[row,active]=(weak<0.5).astype(np.float32);mask[row,active]=1.0
    return target,mask

class VisibilityDataset(Dataset):
    def __init__(self,path,indices):
        self.base=NineClassDataset(path,indices,label_indices=np.asarray(SELECTED_INDICES,dtype=np.int64));self.target,self.mask=visibility_targets(path,indices)
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x=self.base[i];x['visibility_target']=torch.from_numpy(self.target[i]);x['visibility_mask']=torch.from_numpy(self.mask[i]);return x
    def close(self):self.base.close()

class AuxModel(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=build_backbone('dfms_iq',num_classes=6,width=16,dropout=0.2);self.visibility_head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,6));self.residual_bound=0.25
    def forward(self,x,return_aux=False):
        logits,features=self.backbone(x,return_features=True);aux=self.visibility_head(features)
        return (logits,aux) if return_aux else logits

VARIANTS=('F2_BCE','F2_BCE_visibility');SEEDS=(2039,2040,2041)
def set_seed(s):np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def predict(model,path,indices,device,batch,include_visibility=False):
    ds=VisibilityDataset(path,indices);dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True);ys=[];ls=[];vs=[];vts=[];vms=[]
    try:
        model.eval()
        with torch.no_grad():
            for b in dl:
                out=model(b['signal'].to(device,non_blocking=True),return_aux=include_visibility)
                if include_visibility:logits,aux=out;vs.append(aux.cpu().numpy());vts.append(b['visibility_target'].numpy());vms.append(b['visibility_mask'].numpy())
                else:logits=out
                ls.append(logits.cpu().numpy());ys.append(b['label'].numpy())
    finally:ds.close()
    result=(np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32))
    if include_visibility:result+=(np.concatenate(vs).astype(np.float32),np.concatenate(vts).astype(np.float32),np.concatenate(vms).astype(np.float32))
    return result
def run_one(args,name,seed,fit,val,power_dev,device,out):
    set_seed(seed);model=AuxModel().to(device);train_path=find_file(args.data_root,30,'train');ds=VisibilityDataset(train_path,fit);dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay);cls_loss=nn.BCEWithLogitsLoss();hist=[];best=-1;best_epoch=0;started=time.time()
    try:
        for ep in range(1,args.epochs+1):
            model.train();losses=[];closses=[];alosses=[]
            for b in dl:
                x=b['signal'].to(device,non_blocking=True);y=b['label'].to(device,non_blocking=True);opt.zero_grad(set_to_none=True);logits,aux=model(x,return_aux=True);lc=cls_loss(logits,y)
                if name=='F2_BCE_visibility':
                    m=b['visibility_mask'].to(device,non_blocking=True);t=b['visibility_target'].to(device,non_blocking=True);la=nn.functional.binary_cross_entropy_with_logits(aux[m.bool()],t[m.bool()]) if bool(m.bool().any()) else aux.sum()*0;loss=lc+args.aux_weight*la
                else:la=aux.sum()*0;loss=lc
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach().cpu()));closses.append(float(lc.detach().cpu()));alosses.append(float(la.detach().cpu()))
            yt,yl,va,vt,vm=predict(model,train_path,val,device,args.eval_batch_size,True);vmtr=metrics(yt,yl);vaux=float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(va[vm>0]),torch.from_numpy(vt[vm>0])))
            hist.append({'epoch':ep,'train_loss':float(np.mean(losses)),'train_cls_loss':float(np.mean(closses)),'train_aux_loss':float(np.mean(alosses)),'validation':vmtr,'validation_visibility_bce':vaux})
            if vmtr['macro_aupr']>best:best=vmtr['macro_aupr'];best_epoch=ep;torch.save(model.state_dict(),out/f'{name}__seed_{seed}.pth')
    finally:ds.close()
    model.load_state_dict(torch.load(str(out/f'{name}__seed_{seed}.pth'),map_location=device,weights_only=True));r={'variant':name,'seed':seed,'history':hist,'best_epoch':best_epoch,'validation':hist[best_epoch-1]['validation'],'validation_visibility_bce':hist[best_epoch-1]['validation_visibility_bce'],'power_dev':{},'total_parameters':sum(p.numel() for p in model.parameters()),'cost_profile':profile_cost(model,device),'aux_weight':args.aux_weight,'locked_test_used':False,'elapsed_seconds':time.time()-started}
    for p,(path,idx) in power_dev.items():yt,yl,va,vt,vm=predict(model,path,idx,device,args.eval_batch_size,True);r['power_dev'][str(p)]=metrics(yt,yl);r['power_dev'][str(p)]['visibility_bce']=float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(va[vm>0]),torch.from_numpy(vt[vm>0])))
    r['checkpoint_sha256']=sha256(out/f'{name}__seed_{seed}.pth');write_json(out/f'{name}__seed_{seed}.json',r);print(json.dumps({'variant':name,'seed':seed,'val_macro_aupr':r['validation']['macro_aupr']},ensure_ascii=False),flush=True);return r
def run(args):
    device=torch.device(args.device)
    if device.type!='cuda':raise RuntimeError('P2c requires CUDA')
    out=args.output_root;out.mkdir(parents=True,exist_ok=True);base=find_file(args.data_root,30,'train');pure=pure_six_indices(base);perm=np.random.default_rng(args.split_seed).permutation(pure);fit=np.sort(perm[:args.fit_samples]);val=np.sort(perm[args.fit_samples:args.fit_samples+args.val_samples]);used=set(receiver_seeds(base,fit));power_dev={}
    for p in POWERS_DB:
        path=find_file(args.data_root,p,'train');pp=pure_six_indices(path)
        with h5py.File(str(path),'r') as h:rs=np.asarray(h['receiver_noise_seed'][:,:],dtype=np.int64).reshape(-1)
        cand=pp[~np.isin(rs[pp],np.fromiter(used,dtype=np.int64))];idx=np.sort(np.random.default_rng(args.split_seed+p).choice(cand,size=min(args.power_dev_samples,len(cand)),replace=False));power_dev[p]=(path,idx)
    role={'schema':'route-a-p2c-visibility-aux-v1','selected_classes':list(CLASS_NAMES),'weak_mask_alignment':'active selected labels in ascending global label index','fit':{'path':str(base),'count':len(fit),'sample_indices':fit.tolist(),'receiver_noise_seed':receiver_seeds(base,fit)},'validation':{'path':str(base),'count':len(val),'sample_indices':val.tolist(),'receiver_noise_seed':receiver_seeds(base,val)},'power_development':{'source_role':'train_only_unseen_identity','count_by_power':{str(p):len(i) for p,(path,i) in power_dev.items()}},'test':{'used':False,'locked_test_used':False}};write_json(out/'role_manifest.json',role)
    rows=[]
    for name in VARIANTS:
        d=out/name;d.mkdir(parents=True,exist_ok=True)
        for seed in args.seeds:rows.append(run_one(args,name,int(seed),fit,val,power_dev,device,d))
    summary={'schema':'route-a-p2c-visibility-aux-summary-v1','status':'completed','locked_test_used':False,'variants':{}}
    for name in VARIANTS:
        c=[r for r in rows if r['variant']==name];summary['variants'][name]={'parameters':c[0]['total_parameters'],'val_macro_aupr_mean':float(np.mean([r['validation']['macro_aupr'] for r in c])),'val_macro_aupr_std':float(np.std([r['validation']['macro_aupr'] for r in c],ddof=1)),'val_combo_mean':float(np.mean([r['validation']['combination_exact_accuracy'] for r in c])),'val_visibility_bce_mean':float(np.mean([r['validation_visibility_bce'] for r in c])),'power_dev_macro_aupr_mean_by_power':{str(p):float(np.mean([r['power_dev'][str(p)]['macro_aupr'] for r in c])) for p in POWERS_DB}}
    write_json(out/'p2c_visibility_aux_summary.json',summary);write_json(out/'terminal_receipt.json',{'status':'completed','summary':str((out/'p2c_visibility_aux_summary.json').resolve()),'locked_test_used':False,'test_files_opened':False,'script_sha256':sha256(Path(__file__))});print(json.dumps({'status':'completed','summary':str(out/'p2c_visibility_aux_summary.json')},ensure_ascii=False),flush=True)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path(r'E:\论文相关\干扰数据\复合干扰数据0716\九类'));p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/route_a_p2c_visibility_aux_20260827');p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));p.add_argument('--split-seed',type=int,default=260827);p.add_argument('--fit-samples',type=int,default=4096);p.add_argument('--val-samples',type=int,default=2048);p.add_argument('--power-dev-samples',type=int,default=512);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--eval-batch-size',type=int,default=32);p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--aux-weight',type=float,default=0.1);p.add_argument('--device',default='cuda:0');return p.parse_args()
if __name__=='__main__':run(parse_args())
