function receipt = generate_route_a_v7_dataset(executeGeneration)
%GENERATE_ROUTE_A_V7_DATASET Generate train-only fresh development-v7 data.
    if nargin < 1
        executeGeneration = false;
    end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    codeRoot = fileparts(projectRoot);
    configHits = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
    if isempty(configHits)
        error('RouteA:GeneratorNotFound','Cannot locate the closed-set generator.');
    end
    generatorRoot = configHits(1).folder;
    dataRoot = fileparts(fileparts(generatorRoot));
    v6Hits = dir(fullfile(dataRoot, '*0801_v6'));
    v6Hits = v6Hits([v6Hits.isdir]);
    if numel(v6Hits) ~= 1
        error('RouteA:DataRootAmbiguous','Expected exactly one development-v6 directory.');
    end
    outputRoot = fullfile(dataRoot, strrep(v6Hits(1).name, 'v6', 'v7'));
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_development_v7_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg = create_closedset_6class_config();
    cfg.seed = 74000000;
    cfg.output_dir = outputRoot;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), 'train', 'resume');
    count = double(cfg.dataset.num_samples);
    current = [double(cfg.seed)+1, double(cfg.seed)+count];
    references = [20260732,20285931;23260732,23285931;24000001,24025200; ...
        27000001,27025200;34000001,34025200;37000001,37025200; ...
        44000001,44025200;47000001,47025200;54000001,54025200; ...
        57000001,57025200;64000001,64005040];
    for index = 1:size(references,1)
        if ~(current(2) < references(index,1) || current(1) > references(index,2))
            error('RouteA:IdentityOverlap','development-v7 overlaps a prior role.');
        end
    end
    receipt = struct('status','completed','executed',logical(executeGeneration), ...
        'base_seed',cfg.seed,'output_dir',cfg.output_dir,'split_selection','train_only', ...
        'train_path',result.train_path,'train_status',result.splits.train.status, ...
        'evidence_role','development_v7','test_v7_generated',false, ...
        'development_v7_sample_seed_range',current, ...
        'reserved_test_v7_sample_seed_range',[double(cfg.seed)+3000001,double(cfg.seed)+3000000+count], ...
        'identity_disjoint_precheck',true,'locked_test_used',false, ...
        'test_v2_used',false,'test_v3_used',false,'test_v7_used',false);
    if executeGeneration
        receipt.train_summary = result.splits.train.summary;
    end
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix = 'dry_run'; if executeGeneration; prefix='formal'; end
    receiptPath = fullfile(auditRoot,sprintf('%s_train_generation_receipt.json',prefix));
    fid=fopen(receiptPath,'w'); cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
    payload = unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8');
    fwrite(fid,payload,'uint8');
    fprintf('ROUTE_A_V7_RECEIPT=%s\n',receiptPath);
end
