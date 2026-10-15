function receipt = generate_route_a_development_v15_pamiq_positional_query(executeGeneration)
%GENERATE_ROUTE_A_DEVELOPMENT_V15_PAMIQ_POSITIONAL_QUERY Generate train only.
% The reserved test role is deliberately not generated or opened.
    if nargin < 1; executeGeneration = false; end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    codeRoot = fileparts(projectRoot);
    configHits = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
    if isempty(configHits); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
    generatorRoot = configHits(1).folder;
    dataRoot = fileparts(fileparts(generatorRoot));
    outputRoot = fullfile(dataRoot, '复合干扰数据0802_v15_pamiq_positional_query');
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_development_v15_pamiq_positional_query_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg = create_closedset_6class_config();
    cfg.seed = 450000000;
    cfg.output_dir = outputRoot;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg); schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), 'train', 'resume');
    count = double(cfg.dataset.num_samples); current = [double(cfg.seed)+1, double(cfg.seed)+count];
    if ~isequal(current,[450000001,450025200]); error('RouteA:IdentityContract','Unexpected v15 sample range.'); end
    receipt = struct('status','completed','executed',logical(executeGeneration), ...
        'evidence_role','development_v15_pamiq_positional_query','base_seed',cfg.seed, ...
        'output_dir',cfg.output_dir,'split_selection','train_only','train_path',result.train_path, ...
        'train_status',result.splits.train.status,'development_v15_sample_seed_range',current, ...
        'development_v15_receiver_noise_seed_range',[648677743,648702942], ...
        'test_v15_generated',false,'test_files_opened',false,'locked_test_used',false, ...
        'test_v2_used',false,'test_v3_used',false,'test_v15_used',false);
    if executeGeneration; receipt.train_summary=result.splits.train.summary; end
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix='dry_run'; if executeGeneration; prefix='formal'; end
    receiptPath=fullfile(auditRoot,sprintf('%s_train_generation_receipt.json',prefix));
    fid=fopen(receiptPath,'w'); cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8');
    fprintf('ROUTE_A_V15_PAMIQ_POSITIONAL_QUERY_RECEIPT=%s\n',receiptPath);
end
