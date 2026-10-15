function receipt = generate_route_a_v2_dataset(executeGeneration, splitSelection)
%GENERATE_ROUTE_A_V2_DATASET Generate identity-separated development-v2/test-v2.
% The physical configuration is inherited from the audited 0716 generator;
% only the base seed and output root are changed. The test-v2 file is generated
% and verified by the generator, but downstream experiments must keep it sealed.

    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        splitSelection = 'both';
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0731_v2';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', ...
        'route_a_development_v2_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    cfg = create_closedset_6class_config();
    cfg.seed = 20260731;
    cfg.output_dir = outputRoot;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, ...
        logical(executeGeneration), splitSelection, 'resume');

    receipt = struct();
    receipt.status = 'completed';
    receipt.executed = logical(executeGeneration);
    receipt.base_seed = cfg.seed;
    receipt.output_dir = cfg.output_dir;
    receipt.split_selection = splitSelection;
    receipt.train_path = result.train_path;
    receipt.test_path = result.test_path;
    receipt.development_v2_role = 'fit_early_calibration_threshold_only';
    receipt.test_v2_role = 'sealed_not_for_selection';
    receipt.locked_test_used = false;
    receipt.train_status = result.splits.train.status;
    receipt.test_status = result.splits.test.status;
    receipt.train_summary = result.splits.train.summary;
    receipt.test_summary = result.splits.test.summary;
    receipt.generated_at = char(datetime('now', ...
        'TimeZone', 'Asia/Shanghai', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));

    if ~isfolder(auditRoot)
        mkdir(auditRoot);
    end
    if executeGeneration
        receiptName = 'formal_generation_receipt.json';
    else
        receiptName = 'dry_run_receipt.json';
    end
    receiptPath = fullfile(auditRoot, receiptName);
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0
        error('RouteA:ReceiptWriteFailed', 'Cannot write %s', receiptPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_V2_RECEIPT=%s\n', receiptPath);
end

