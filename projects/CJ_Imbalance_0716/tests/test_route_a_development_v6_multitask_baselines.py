import torch
from experiments.run_route_a_development_v6_multitask_baselines import _cagrad_weight,_dot,_mgda_weight,_pcgrad_two,_validate_v6_config

def _valid(): return {"data_root":r"C:\data\v6","train_file":"train.mat","test_v6_path":r"C:\data\v6\test.mat","sealed_test_v3_path":r"C:\data\v3\test.mat","forbidden_test_v2_path":r"C:\data\v2\test.mat","allow_locked_test":False,"evidence_role":"development_v6","test_v6_generated":False,"test_v6_access_authorized":False,"test_v6_used":False,"test_v3_sealed":True,"test_v3_used":False,"test_v2_consumed":True,"development_v5_reuse_for_selection":False}
def test_v6_config_guards_all_frozen_roles():
    _validate_v6_config(_valid())
    for update in ({"allow_locked_test":True},{"evidence_role":"development_v5"},{"test_v6_generated":True},{"test_v6_access_authorized":True},{"test_v6_used":True},{"test_v3_sealed":False},{"test_v3_used":True},{"test_v2_consumed":False},{"development_v5_reuse_for_selection":True},{"train_file":"test.mat"}):
        candidate=_valid(); candidate.update(update)
        try: _validate_v6_config(candidate)
        except ValueError: pass
        else: raise AssertionError('invalid v6 config accepted')
def test_pcgrad_removes_two_task_conflict():
    first=[torch.tensor([1.0,0.0])]; second=[torch.tensor([-1.0,1.0])]; combined,projected=_pcgrad_two(first,second)
    assert projected; assert torch.isfinite(combined[0]).all(); assert float(_dot(combined,combined))>0
def test_mgda_weight_is_bounded_and_minimizes_two_vector_norm():
    first=[torch.tensor([1.0,0.0])]; second=[torch.tensor([0.0,2.0])]; weight=_mgda_weight(first,second)
    assert 0<=weight<=1
    norm=lambda w: float(((w*first[0]+(1-w)*second[0])**2).sum())
    assert norm(weight)<=min(norm(0.0),norm(1.0))+1e-7
def test_cagrad_grid_weight_is_deterministic_and_bounded():
    first=[torch.tensor([1.0,0.0])]; second=[torch.tensor([-0.5,1.0])]
    left=_cagrad_weight(first,second,c=0.4,grid_size=101); right=_cagrad_weight(first,second,c=0.4,grid_size=101)
    assert left==right and 0<=left<=1
