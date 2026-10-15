function receipt = generate_route_a_v3_power_sweep(executeGeneration, levels_dB)
%GENERATE_ROUTE_A_V3_POWER_SWEEP Identity-paired train-only power views for development-v3.
% Test power views are deliberately unsupported so the sealed test-v3 cannot
% acquire an alternate analysis path during model development.

    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        levels_dB = 10:5:50;
    end
    validatestring(string(executeGeneration), ["0", "1", "false", "true"]);
    levels_dB = reshape(double(levels_dB), 1, []);
    standardLevels = 10:5:50;
    if isempty(levels_dB) || numel(unique(levels_dB)) ~= numel(levels_dB) || ...
            ~all(ismember(levels_dB, standardLevels))
        error('RouteA:InvalidPowerLevels', ...
            'levels_dB must be a unique nonempty subset of 10:5:50.');
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    sweepRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0731_v3\power_sweep_9level_v3';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', ...
        'route_a_v3_power_sweep_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    configs = cell(numel(levels_dB), 1);
    for index = 1:numel(levels_dB)
        level = levels_dB(index);
        cfg = create_power_sweep_6class_config(level);
        cfg.seed = 24000000;
        levelIndex = find(standardLevels == level, 1, 'first');
        cfg.output_dir = fullfile(sweepRoot, sprintf( ...
            'level_%02d__jnr_%ddB', levelIndex, level));
        configs{index} = cfg;
    end

    result = run_power_sweep_levels(configs, logical(executeGeneration), ...
        'train', 'resume', logical(executeGeneration));

    receipt = struct();
    receipt.status = 'completed';
    receipt.executed = logical(executeGeneration);
    receipt.base_seed = 24000000;
    receipt.levels_dB = levels_dB;
    receipt.split_selection = 'train_only';
    receipt.sweep_root = sweepRoot;
    receipt.manifest_path = result.manifest_path;
    receipt.expected_samples_per_level = 25200;
    receipt.receiver_identity_contract = ...
        'same receiver_noise_seed and labels across levels; roles inherited from development-v3';
    receipt.test_power_views_generated = false;
    receipt.locked_test_used = false;
    receipt.test_v2_used = false;
    receipt.test_v3_used = false;
    receipt.generated_at = char(datetime('now', 'TimeZone', ...
        'Asia/Shanghai', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    if ~isfolder(auditRoot)
        mkdir(auditRoot);
    end
    prefix = 'dry_run';
    if executeGeneration
        prefix = 'formal';
    end
    receiptPath = fullfile(auditRoot, sprintf( ...
        '%s_power_sweep_generation_receipt.json', prefix));
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0
        error('RouteA:ReceiptWriteFailed', 'Cannot write %s', receiptPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_V3_POWER_SWEEP_RECEIPT=%s\n', receiptPath);
end

