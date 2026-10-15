function receipt = generate_route_a_v3_dataset(executeGeneration, splitSelection)
%GENERATE_ROUTE_A_V3_DATASET Generate fresh development-v3/test-v3 data.
% The historical locked test and consumed test-v2 are never opened.  The base
% seed is deliberately separated from both development-v2 train and test seed
% intervals by more than one complete split.  Downstream code must seal the
% test-v3 file immediately and may only use the train split for development.

    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        splitSelection = 'train';
    end
    splitSelection = validatestring(splitSelection, {'train', 'test', 'both'});

    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0731_v3';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', ...
        'route_a_development_v3_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    cfg = create_closedset_6class_config();
    cfg.seed = 24000000;
    cfg.output_dir = outputRoot;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, ...
        logical(executeGeneration), splitSelection, 'resume');

    sampleCount = double(cfg.dataset.num_samples);
    v3Range = [double(cfg.seed) + 1, double(cfg.seed) + sampleCount];
    testV3Range = [double(cfg.seed) + 3000000 + 1, ...
        double(cfg.seed) + 3000000 + sampleCount];
    v2BaseSeed = 20260731;
    v2TrainRange = [v2BaseSeed + 1, v2BaseSeed + sampleCount];
    v2TestRange = [v2BaseSeed + 3000000 + 1, ...
        v2BaseSeed + 3000000 + sampleCount];
    disjointTrain = v3Range(2) < v2TrainRange(1) || ...
        v3Range(1) > v2TrainRange(2);
    disjointTest = v3Range(2) < v2TestRange(1) || ...
        v3Range(1) > v2TestRange(2);
    if ~(disjointTrain && disjointTest)
        error('RouteA:IdentityOverlap', ...
            'development-v3 sample_seed range overlaps v2 train/test.');
    end

    receipt = struct();
    receipt.status = 'completed';
    receipt.executed = logical(executeGeneration);
    receipt.base_seed = cfg.seed;
    receipt.output_dir = cfg.output_dir;
    receipt.split_selection = splitSelection;
    receipt.train_path = result.train_path;
    receipt.test_path = result.test_path;
    receipt.train_status = result.splits.train.status;
    receipt.test_status = result.splits.test.status;
    if any(strcmp(splitSelection, {'train', 'both'}))
        receipt.train_summary = result.splits.train.summary;
    end
    receipt.development_v3_role = ...
        'optimization_validation_early_calibration_threshold_only';
    receipt.test_v3_role = 'sealed_not_for_selection_or_analysis';
    receipt.sample_seed_range = v3Range;
    receipt.development_v3_sample_seed_range = v3Range;
    receipt.test_v3_sample_seed_range = testV3Range;
    receipt.reference_v2_train_sample_seed_range = v2TrainRange;
    receipt.reference_v2_test_sample_seed_range = v2TestRange;
    receipt.identity_disjoint_precheck = true;
    receipt.receiver_noise_seed_mapping = ...
        'bijective_additive_uint32_mapping_from_sample_seed';
    receipt.sealed_test_v2_path = ...
        'C:\LJQ_code\干扰数据\复合干扰数据0731_v2\closedset_6class_test.mat';
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
    receiptName = sprintf('%s_%s_generation_receipt.json', ...
        executionPrefix, splitSelection);
    receiptPath = fullfile(auditRoot, receiptName);
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0
        error('RouteA:ReceiptWriteFailed', 'Cannot write %s', receiptPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_V3_RECEIPT=%s\n', receiptPath);
end
