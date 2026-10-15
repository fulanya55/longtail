"""Audit the fresh v23 source using the fixed three-stratum identity contract."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
from experiments.audit_route_a_v22_visibility_balanced_morphology_expert import audit
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path',type=Path,default=Path(r'C:\LJQ_code\干扰数据\复合干扰数据0802_v23_weak_pair_ranking_morphology\closedset_6class_train.mat'))
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v23_weak_pair_ranking_morphology_generation/identity_audit.json')
    parser.add_argument('--roles-output',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_v23_weak_pair_ranking_morphology_generation/role_indices.npz')
    args=parser.parse_args(); payload,roles=audit(args.data_path); payload['evidence_role']='fresh_v23_weak_pair_ranking_morphology_expert'; payload['test_v23_used']=False
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    import numpy as np
    np.savez_compressed(args.roles_output,**roles);print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
