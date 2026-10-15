function receipt = generate_route_a_v5_dataset(executeGeneration)
%GENERATE_ROUTE_A_V5_DATASET Generate train-only fresh development-v5 data.

    if nargin < 1
        executeGeneration = false;
    end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0801_v5';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', ...
        'route_a_development_v5_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    cfg = create_closedset_6class_config();
    cfg.seed = 44000000;
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
    ranges.development_v4_train = [34000001, 34025200];
    ranges.reserved_test_v4 = [37000001, 37025200];
    ranges.development_v5_train = [double(cfg.seed) + 1, ...
        double(cfg.seed) + sampleCount];
    ranges.reserved_test_v5 = [double(cfg.seed) + 3000000 + 1, ...
        double(cfg.seed) + 3000000 + sampleCount];
    references = {'development_v2_train', 'test_v2', ...
        'development_v3_train', 'sealed_test_v3', ...
        'development_v4_train', 'reserved_test_v4'};
    for index = 1:numel(references)
        current = ranges.development_v5_train;
        reference = ranges.(references{index});
        if ~(current(2) < reference(1) || current(1) > reference(2))
            error('RouteA:IdentityOverlap', ...
                'development-v5 overlaps %s.', references{index});
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
    receipt.evidence_role = 'development_v5';
    receipt.test_v5_generated = false;
    receipt.identity_seed_ranges = ranges;
    receipt.identity_disjoint_precheck = true;
    receipt.locked_test_used = false;
    receipt.test_v2_used = false;
    receipt.test_v3_used = false;
    receipt.test_v4_used = false;
    receipt.test_v5_used = false;
    receipt.generated_at = char(datetime('now', ...
        'TimeZone', 'Asia/Shanghai', 'Format', ...
        'yyyy-MM-dd''T''HH:mm:ssXXX'));

    if ~isfolder(auditRoot)
        mkdir(auditRoot);
    end
    prefix = 'dry_run';
    if executeGeneration
        prefix = 'formal';
    end
    receiptPath = fullfile(auditRoot, ...
        sprintf('%s_train_generation_receipt.json', prefix));
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0
        error('RouteA:ReceiptWriteFailed', 'Cannot write %s', receiptPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_V5_RECEIPT=%s\n', receiptPath);
end

