function receipt = generate_route_a_h0s_isrj_combined_base(executeGeneration)
%GENERATE_ROUTE_A_H0S_ISRJ_COMBINED_BASE Train-only high-replica base.
if nargin < 1; executeGeneration = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
codeRoot = fileparts(projectRoot);
configHits = dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(configHits); error('RouteA:GeneratorNotFound','Cannot locate closed-set generator.'); end
generatorRoot = configHits(1).folder;
dataRoot = fileparts(fileparts(generatorRoot));
outputRoot = fullfile(dataRoot,'复合干扰数据0803_isrj_ood_h0s_r1_ar1_base');
auditRoot = fullfile(projectRoot,'outputs','tapc','route_a_h0s_isrj_combined_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg = create_closedset_6class_config();
cfg.seed = 810000000;
cfg.output_dir = outputRoot;
cfg.output.save_legacy = false;
cfg.dataset.samples_per_pattern = 80;
cfg.dataset.num_samples = 5040;
cfg.params.ISRJ.sample_width_s = [25e-6,30e-6];
cfg.params.ISRJ.num_replicas = [9,9];
cfg = normalize_and_validate_config(cfg,'balanced');
registry = build_jamming_registry(cfg);
schema = build_parameter_schema(registry);
result = run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
receipt = struct('status','completed','executed',logical(executeGeneration), ...
    'experiment','RouteA_H0s_ISRJCombinedParameterNoiseFrozen', ...
    'evidence_role','fresh_joint_parameter_receiver_noise_frozen_replication', ...
    'base_seed',cfg.seed,'output_dir',outputRoot,'split_selection','train_only', ...
    'train_path',result.train_path,'train_status',result.splits.train.status, ...
    'sample_count',5040,'samples_per_pattern',80, ...
    'isrj_sample_width_s',cfg.params.ISRJ.sample_width_s, ...
    'isrj_num_replicas',cfg.params.ISRJ.num_replicas, ...
    'expected_receiver_noise_seed_range',[1008677743,1008682782], ...
    'test_split_generated',false,'test_files_opened',false,'locked_test_used',false, ...
    'test_v2_used',false,'test_v3_used',false,'test_h0s_used',false);
if executeGeneration; receipt.train_summary = result.splits.train.summary; end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
receiptPath=fullfile(auditRoot,[prefix '_base_generation_receipt.json']);
fid=fopen(receiptPath,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H0S_BASE_RECEIPT=%s\n',receiptPath);
end
