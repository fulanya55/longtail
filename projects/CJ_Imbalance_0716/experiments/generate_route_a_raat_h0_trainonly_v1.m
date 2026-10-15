function receipt = generate_route_a_raat_h0_trainonly_v1(executeGeneration)
%GENERATE_ROUTE_A_RAAT_H0_TRAINONLY_V1 Frozen RAAT-H0/v1 train source only.
% This function never requests, creates, or opens a test split.
if nargin < 1; executeGeneration = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
codeRoot = fileparts(projectRoot);
configHits = dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(configHits); error('RouteA:GeneratorNotFound','Cannot locate closed-set generator.'); end
generatorRoot = configHits(1).folder;
dataRoot = fileparts(fileparts(generatorRoot));
outputRoot = fullfile(dataRoot,'复合干扰数据0802_raat_h0_v1');
auditRoot = fullfile(projectRoot,'outputs','tapc','route_a_raat_h0_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));

cfg = create_closedset_6class_config();
cfg.seed = 780000000;
cfg.output_dir = outputRoot;
cfg.output.save_legacy = false;
cfg.dataset.samples_per_pattern = 80;
cfg.dataset.num_samples = 5040;
cfg = normalize_and_validate_config(cfg,'balanced');
registry = build_jamming_registry(cfg);
schema = build_parameter_schema(registry);
result = run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');

receipt = struct('status','completed','executed',logical(executeGeneration), ...
    'experiment','RouteA_RAAT_H0_v1','evidence_role','fresh_train_only_physical_identifiability', ...
    'base_seed',cfg.seed,'output_dir',outputRoot,'split_selection','train_only', ...
    'train_path',result.train_path,'train_status',result.splits.train.status, ...
    'sample_count',5040,'samples_per_pattern',80, ...
    'expected_receiver_noise_seed_range',[978677743,978682782], ...
    'generator_change','none; inherited 0716 balanced six-label protocol', ...
    'test_split_generated',false,'test_files_opened',false,'locked_test_used',false, ...
    'test_v2_used',false,'test_v3_used',false,'test_raat_used',false);
if executeGeneration; receipt.train_summary = result.splits.train.summary; end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
receiptPath=fullfile(auditRoot,[prefix '_generation_receipt.json']);
fid=fopen(receiptPath,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_RAAT_H0_RECEIPT=%s\n',receiptPath);
end
