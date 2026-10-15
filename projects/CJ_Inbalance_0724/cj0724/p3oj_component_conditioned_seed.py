"""One durable P3o-J seed: J0 freezes, J1 trains posterior component correction."""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
import torch
import cj0724.p3og_bounded_combo_cvar as core
from cj0724.p3oi_durable_finite_combo_cvar_seed import finite_valid_per_sample_parameter_loss
from cj0724.p3oj_component_conditioned_residual import ComponentConditionedResidual
from cj0724.p3o_low_power_attribution import audit_identities as audit

def _audit(project, train, development, ignored, original):
    return audit(project, train, development, {1450000000,1460000000}, original)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-seed',type=int,choices=(2091,2092,2093),required=True); p.add_argument('--output-root',required=True); a=p.parse_args()
    out=Path(__file__).resolve().parents[1]/a.output_root
    core.BoundedStatePreservingResidual=ComponentConditionedResidual; core._per_sample_parameter_loss=finite_valid_per_sample_parameter_loss; core.audit_identities=_audit; core.MODEL_SEEDS=(a.model_seed,)
    original=core._build_side
    def build(base,state,bound,device):
        return original(base,state,bound,device)
    core._build_side=build
    original_epoch=core.train_side_epoch
    def epoch(model,*args):
        # The legacy method flag only selects J0/J1; both use mean loss.
        model.set_component_enabled(args[-1] > 0.0)
        return original_epoch(model,*args[:-1],0.0)
    core.train_side_epoch=epoch
    sys.argv=[sys.argv[0],'--train-root-template','generated_data/p3oj_component_conditioned_train_seed1450000000_{regime}_rechunked','--development-root-template','generated_data/p3oj_component_conditioned_dev_seed1460000000_{regime}_rechunked','--preregister','configs/p3oj_component_conditioned_residual_preregister_20260802.json','--method-freeze','configs/p3oj_component_conditioned_residual_method_freeze_20260802.json','--output-root',a.output_root]
    try: core.main()
    except Exception as e:
        out.mkdir(parents=True,exist_ok=True); (out/'exception_receipt.json').write_text(json.dumps({'model_seed':a.model_seed,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'test_generated':False,'test_opened':False,'locked_test_used':False},indent=2),encoding='utf-8'); raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
if __name__=='__main__': main()
