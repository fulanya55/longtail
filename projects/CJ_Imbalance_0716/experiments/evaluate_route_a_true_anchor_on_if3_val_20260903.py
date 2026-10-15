"""Evaluate independently trained true Anchor checkpoints on the IF3 validation split."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from cjlt.models.backbone_adapter import build_backbone
from experiments.run_route_a_ama_if3_anchor_baseline_20260903 import DATA_ROOT, OUT, ArrayDataset, load_mat, split_by_combo
from experiments.run_route_a_c1_baseline import initialize_f2, metrics
TRUE_OUT=ROOT/'outputs'/'route_a_ama_true_anchor_20260903'
def pred(model, signal, labels, idx, device):
 ds=ArrayDataset(signal,labels,idx); dl=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,pin_memory=True); ys=[];ls=[]; model.eval()
 with torch.no_grad():
  for x,y in dl: ys.append(y.numpy());ls.append(model(x.to(device,non_blocking=True)).cpu().numpy())
 return metrics(np.concatenate(ys).astype(np.uint8),np.concatenate(ls).astype(np.float32))
def main():
 device=torch.device('cuda:0')
 train_signal,train_labels,_=load_mat(DATA_ROOT/'closedset_6class_train.mat')
 _,val_idx=split_by_combo(train_labels,seed=20260903)
 rows=[]
 for seed in (2039,2040,2041):
  model=build_backbone('dfms_iq',num_classes=6,width=16,dropout=0.2).to(device)
  initialize_f2(model,Path(r'E:\LJQ_code\CJ_Imbalance_0716\frozen\20260731_route_a_visibility_balanced_fusion_v2\seed_2039\backbone\best_model.pth'))
  ck=TRUE_OUT/f'true_anchor__seed_{seed}.pth'; model.load_state_dict(torch.load(str(ck),map_location=device,weights_only=False)['model_state_dict'])
  row={'seed':seed,'if3_validation':pred(model,train_signal,train_labels,val_idx,device),'locked_test_used':False,'frozen_e5_modified':False}; rows.append(row)
  print(json.dumps({'seed':seed,'macro_aupr':row['if3_validation']['macro_aupr'],'macro_f1':row['if3_validation']['macro_f1'],'combo':row['if3_validation']['combination_exact_accuracy']},ensure_ascii=False),flush=True)
 out=TRUE_OUT/'on_if3_validation'; out.mkdir(parents=True,exist_ok=True); s={'schema':'route-a-ama-true-anchor-on-if3-validation-v1','status':'completed','split_seed':20260903,'validation_indices':int(len(val_idx)),'runs':rows,'locked_test_used':False,'frozen_e5_modified':False,'evidence_boundary':'development-validation-only'}; (out/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'status':'completed','summary':str((out/'summary.json').resolve())}),flush=True)
if __name__=='__main__': main()
