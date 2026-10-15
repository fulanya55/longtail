function receipt = generate_route_a_development_v16_isrj_width_probe(executeGeneration)
%GENERATE_ROUTE_A_DEVELOPMENT_V16_ISRJ_WIDTH_PROBE Generate only a fresh train role.
    if nargin < 1; executeGeneration = false; end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    codeRoot = fileparts(projectRoot);
    configHits = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
    if isempty(configHits); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
    generatorRoot = configHits(1).folder;
    dataRoot = fileparts(fileparts(generatorRoot));
    outputRoot = fullfile(dataRoot, '复合干扰数据0802_v16_isrj_width_probe');
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_development_v16_isrj_width_probe_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg = create_closedset_6class_config();
    cfg.seed = 520000000;
    cfg.output_dir = outputRoot;
    cfg.dataset.samples_per_pattern = 80;
    cfg.dataset.num_samples = 5040;
    cfg.params.ISRJ.sample_width_s = [10e-6,30e-6];
    cfg.params.ISRJ.num_replicas = [3,8];
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg); schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), 'train', 'resume');
    count = double(cfg.dataset.num_samples); current = [double(cfg.seed)+1, double(cfg.seed)+count];
    if ~isequal(current,[520000001,520005040]); error('RouteA:IdentityContract','Unexpected v16 sample range.'); end
    receipt = struct('status','completed','executed',logical(executeGeneration), ...
        'evidence_role','development_v16_isrj_width_identifiability','base_seed',cfg.seed, ...
        'output_dir',cfg.output_dir,'split_selection','train_only','train_path',result.train_path, ...
        'train_status',result.splits.train.status,'sample_count',count,'samples_per_pattern',80, ...
        'sample_seed_range',current,'receiver_noise_seed_range',[718677743,718682782], ...
        'isrj_sample_width_s',[10e-6,30e-6],'isrj_num_replicas',[3,8], ...
        'test_v16_generated',false,'test_files_opened',false,'locked_test_used',false, ...
        'test_v2_used',false,'test_v3_used',false,'test_v16_used',false);
    if executeGeneration; receipt.train_summary=result.splits.train.summary; end
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix='dry_run'; if executeGeneration; prefix='formal'; end
    receiptPath=fullfile(auditRoot,sprintf('%s_train_generation_receipt.json',prefix));
    fid=fopen(receiptPath,'w'); cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8');
    fprintf('ROUTE_A_V16_ISRJ_WIDTH_PROBE_RECEIPT=%s\n',receiptPath);
end
