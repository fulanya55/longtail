function receipt = generate_route_a_h0g_isrj_parameter_ood_v2(executeGeneration)
%GENERATE_ROUTE_A_H0G_ISRJ_PARAMETER_OOD_V2 Train-only farther-parameter OOD.
if nargin < 1; executeGeneration = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
codeRoot = fileparts(projectRoot);
configHits = dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(configHits); error('RouteA:GeneratorNotFound','Cannot locate closed-set generator.'); end
generatorRoot = configHits(1).folder;
dataRoot = fileparts(fileparts(generatorRoot));
outputRoot = fullfile(dataRoot,'复合干扰数据0802_isrj_ood_h0g');
auditRoot = fullfile(projectRoot,'outputs','tapc','route_a_h0g_isrj_parameter_ood_v2_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg = create_closedset_6class_config();
cfg.seed = 760000000;
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
    'experiment','RouteA_H0g_ISRJParameterOODv2','evidence_role','fresh_far_parameter_ood_frozen_replication', ...
    'base_seed',cfg.seed,'output_dir',outputRoot,'split_selection','train_only', ...
    'train_path',result.train_path,'train_status',result.splits.train.status, ...
    'sample_count',5040,'samples_per_pattern',80,'isrj_sample_width_s',cfg.params.ISRJ.sample_width_s, ...
    'isrj_num_replicas',cfg.params.ISRJ.num_replicas,'expected_receiver_noise_seed_range',[958677743,958682782], ...
    'test_split_generated',false,'test_files_opened',false,'locked_test_used',false, ...
    'test_v2_used',false,'test_v3_used',false,'test_h0g_used',false);
if executeGeneration; receipt.train_summary = result.splits.train.summary; end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
receiptPath=fullfile(auditRoot,[prefix '_generation_receipt.json']);
fid=fopen(receiptPath,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H0G_RECEIPT=%s\n',receiptPath);
end
