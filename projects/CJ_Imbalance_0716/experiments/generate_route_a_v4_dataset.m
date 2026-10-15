function receipt = generate_route_a_v4_dataset(executeGeneration)
%GENERATE_ROUTE_A_V4_DATASET Generate train-only fresh development-v4 data.
% Historical locked-test, consumed test-v2 and sealed test-v3 files are never
% opened. No test-v4 split is generated during this development round.

    if nargin < 1
        executeGeneration = false;
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0801_v4';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', ...
        'route_a_development_v4_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    cfg = create_closedset_6class_config();
    cfg.seed = 34000000;
    cfg.output_dir = outputRoot;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, ...
        logical(executeGeneration), 'train', 'resume');

    sampleCount = double(cfg.dataset.num_samples);
    ranges = struct();
    ranges.development_v2_train = [20260732, 20285931];
    ranges.test_v2 = [23260732, 23285931];
    ranges.development_v3_train = [24000001, 24025200];
    ranges.sealed_test_v3 = [27000001, 27025200];
    ranges.development_v4_train = [double(cfg.seed) + 1, ...
        double(cfg.seed) + sampleCount];
    ranges.reserved_test_v4 = [double(cfg.seed) + 3000000 + 1, ...
        double(cfg.seed) + 3000000 + sampleCount];

    referenceNames = {'development_v2_train', 'test_v2', ...
        'development_v3_train', 'sealed_test_v3'};
    for index = 1:numel(referenceNames)
        reference = ranges.(referenceNames{index});
        current = ranges.development_v4_train;
        disjoint = current(2) < reference(1) || current(1) > reference(2);
        if ~disjoint
            error('RouteA:IdentityOverlap', ...
                'development-v4 overlaps %s.', referenceNames{index});
        end
    end

    receipt = struct();
    receipt.status = 'completed';
    receipt.executed = logical(executeGeneration);
    receipt.base_seed = cfg.seed;
    receipt.output_dir = cfg.output_dir;
    receipt.split_selection = 'train_only';
    receipt.train_path = result.train_path;
    receipt.train_status = result.splits.train.status;
    if executeGeneration
        receipt.train_summary = result.splits.train.summary;
    end
    receipt.evidence_role = 'development_v4';
    receipt.test_v4_generated = false;
    receipt.identity_seed_ranges = ranges;
    receipt.identity_disjoint_precheck = true;
    receipt.receiver_noise_seed_mapping = ...
        'bijective_additive_uint32_mapping_from_sample_seed';
    receipt.locked_test_used = false;
    receipt.test_v2_used = false;
    receipt.test_v3_used = false;
    receipt.generated_at = char(datetime('now', ...
        'TimeZone', 'Asia/Shanghai', 'Format', ...
        'yyyy-MM-dd''T''HH:mm:ssXXX'));

    if ~isfolder(auditRoot)
        mkdir(auditRoot);
    end
    executionPrefix = 'dry_run';
    if executeGeneration
        executionPrefix = 'formal';
    end
    receiptPath = fullfile(auditRoot, ...
        sprintf('%s_train_generation_receipt.json', executionPrefix));
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0
        error('RouteA:ReceiptWriteFailed', 'Cannot write %s', receiptPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_V4_RECEIPT=%s\n', receiptPath);
end
