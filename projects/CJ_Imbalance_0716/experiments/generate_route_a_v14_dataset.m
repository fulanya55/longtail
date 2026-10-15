function receipt = generate_route_a_v14_dataset(executeGeneration)
%GENERATE_ROUTE_A_V14_DATASET Generate only the preregistered train identity.
    if nargin < 1; executeGeneration = false; end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    codeRoot = fileparts(projectRoot);
    configHits = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
    if isempty(configHits); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
    generatorRoot = configHits(1).folder;
    dataRoot = fileparts(fileparts(generatorRoot));
    outputRoot = fullfile(dataRoot, '复合干扰数据0801_v14');
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_development_v14_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg = create_closedset_6class_config();
    cfg.seed = 144000000;
    cfg.output_dir = outputRoot;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg); schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), 'train', 'resume');
    count = double(cfg.dataset.num_samples); current = [double(cfg.seed)+1, double(cfg.seed)+count];
    if ~isequal(current,[144000001,144025200]); error('RouteA:IdentityContract','Unexpected v14 sample range.'); end
    receipt = struct('status','completed','executed',logical(executeGeneration), ...
        'evidence_role','development_v14','base_seed',cfg.seed,'output_dir',cfg.output_dir, ...
        'split_selection','train_only','train_path',result.train_path, ...
        'train_status',result.splits.train.status,'development_v14_sample_seed_range',current, ...
        'development_v14_receiver_noise_seed_range',[342677743,342702942], ...
        'test_v14_generated',false,'test_files_opened',false,'locked_test_used',false, ...
        'test_v2_used',false,'test_v3_used',false,'test_v14_used',false);
    if executeGeneration; receipt.train_summary=result.splits.train.summary; end
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix='dry_run'; if executeGeneration; prefix='formal'; end
    receiptPath=fullfile(auditRoot,sprintf('%s_train_generation_receipt.json',prefix));
    fid=fopen(receiptPath,'w'); cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8');
    fprintf('ROUTE_A_V14_RECEIPT=%s\n',receiptPath);
end
