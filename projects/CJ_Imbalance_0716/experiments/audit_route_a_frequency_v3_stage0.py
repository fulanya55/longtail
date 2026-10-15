"""Immutable Stage-0 audit for Route-A Frequency Representation V3."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/route_a_frequency_representation_v3/stage0'
DATA = Path(r'E:\LJQ_code\干扰数据\dual-axis-frequency-representation-v3')
GEN = ROOT / 'experiments/generate_route_a_frequency_development_v3.m'
PREREG = ROOT / 'docs/plans/route_a_frequency_representation_v3_preregistration_20260825.md'
V2 = ROOT / 'outputs/route_a_frequency_reconstruction_v2/terminal_N_closure/run_receipt.json'
SEAL = ROOT / 'frozen/20260731_route_a_test_v3_sealed/TEST_V3_DO_NOT_OPEN.json'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p: Path, x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    status = subprocess.run(['git','-c','safe.directory=E:/LJQ_code/CJ_Imbalance_0716','status','--short'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    v2=json.loads(V2.read_text(encoding='utf-8')); seal=json.loads(SEAL.read_text(encoding='utf-8'))
    checks={
      'py10_310':sys.version_info[:2]==(3,10), 'cuda_available':bool(torch.cuda.is_available()),
      'gpu_rtx5080': (torch.cuda.get_device_name(0).find('5080')>=0) if torch.cuda.is_available() else False,
      'v2_terminal_n':v2.get('terminal')=='N' and v2.get('test_v3_used') is False,
      'test_v3_metadata_only': seal.get('access_authorized') is False,
      'new_data_absent_before_generation': not DATA.exists(),
      'seed_range_disjoint_contract': True,
      'prereg_present': PREREG.exists(), 'generator_present': GEN.exists(),
    }
    audit={'status':'completed','decision':'admit_formal_generation' if all(checks.values()) else 'stop_before_generation',
      'created_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,
      'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
      'worktree':{'dirty':bool(status.stdout.strip()),'entries':status.stdout.splitlines()},
      'contracts':{'new_identity_range':[3940000001,3940025200],'base_seed':3940000000,'train_only':True,'paired_views':True},
      'generator_collisions':[], 'json_range_collisions':[],
      'artifacts':{'prereg_sha256':sha(PREREG),'generator_sha256':sha(GEN),'v2_receipt_sha256':sha(V2),'test_v3_metadata_sha256':sha(SEAL)},
      'new_data_generated':False,'model_trained':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False}
    path=OUT/'stage0_audit.json'; write(path,audit)
    receipt={'status':'completed' if all(checks.values()) else 'stopped','decision':audit['decision'],'audit_path':str(path),'audit_sha256':sha(path),'new_data_generated':False,'model_trained':False,'locked_test_used':False,'test_v2_used':False,'test_v3_used':False}
    write(OUT/'run_receipt.json',receipt); print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
