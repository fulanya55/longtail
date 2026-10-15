"""Freeze H1k code prior to fresh data generation."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=('cjlt/models/mft_matched_filter_support.py','experiments/generate_route_a_h1k_mft_matched_filter_composite.m','experiments/audit_route_a_h1k_mft_matched_filter_composite.py','experiments/run_route_a_h1k_mft_matched_filter_composite.py','docs/plans/route_a_h1k_mft_matched_filter_composite_20260802.md','docs/results/route_a_h0j_mft_matched_filter_support_results_20260802.md')
def sha(p):
 h=hashlib.sha256()
 with (ROOT/p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def run():
 gen=(ROOT/FILES[1]).read_text(encoding='utf8').lower();runner=(ROOT/FILES[3]).read_text(encoding='utf8').lower()
 checks={'train_only_generator':"'train','error'" in gen and "test_split_generated',false" in gen,'no_generator_parameter_override':'cfg.params.' not in gen,'fixed_statistic':'effective_delay_support' in runner and 'delay_start' not in runner,'no_training_or_model_head':'adam' not in runner and 'binary_cross_entropy' not in runner and 'torch.nn.linear' not in runner,'evaluation_metadata_after_scores':runner.index('params,_=read_parameters')>runner.index('candidate=_scores')}
 checks['no_threshold_calibration_fusion']='--threshold' not in runner and '--calibration' not in runner and 'platt' not in runner and 'fit_calibration' not in runner and 'fit_fusion' not in runner
 if not all(checks.values()):raise RuntimeError(checks)
 out=ROOT/'outputs/tapc/route_a_h1k_mft_support_freeze/implementation_manifest.json';out.parent.mkdir(parents=True,exist_ok=True);result={'status':'frozen_implementation_before_data_generation','experiment':'RouteA_H1k_MFTMatchedFilterComposite','source_files':{p:sha(p) for p in FILES},'checks':checks,'receiver_noise_seed_range':[1028677743,1028682782],'data_generated':False,'locked_test_used':False,'test_v2_used':False,'test_v3_opened':False,'test_v3_used':False,'next_required_action':'generate_train_only_source_once_then_run_identity_audit'};out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
