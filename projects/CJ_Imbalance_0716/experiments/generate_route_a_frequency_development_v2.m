function receipt = generate_route_a_frequency_development_v2(executeGeneration, levels_dB)
%GENERATE_ROUTE_A_FREQUENCY_DEVELOPMENT_V2 Fresh paired Route-A source.
%   Generates train-only 20/30 dB SawFM views with identical identities.
%   Long-tail frequency manifests are derived later from the 30 dB source;
%   the generator itself remains combination-balanced and never creates test.

    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        levels_dB = [20, 30];
    end
    executeGeneration = logical(executeGeneration);
    levels_dB = reshape(double(levels_dB), 1, []);
    requiredLevels = [20, 30];
    if ~isequal(sort(levels_dB), requiredLevels)
        error('RouteA:FrequencyV2InvalidLevels', ...
            'levels_dB must contain exactly [20, 30].');
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = fullfile('E:\LJQ_code', '干扰数据', '干扰信号生成程序', ...
        '可配置组合干扰数据生成');
    sweepRoot = fullfile('E:\LJQ_code', '干扰数据', ...
        'dual-axis-frequency-development-v2');
    auditRoot = fullfile(projectRoot, 'outputs', ...
        'route_a_frequency_reconstruction_v2', 'stage1_data_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    baseSeed = 3930000000;
    standardLevels = 10:5:50;
    configs = cell(numel(levels_dB), 1);
    for index = 1:numel(levels_dB)
        level = levels_dB(index);
        cfg = create_power_sweep_6class_config(level);
        cfg.seed = baseSeed;
        levelIndex = find(standardLevels == level, 1, 'first');
        cfg.output_dir = fullfile(sweepRoot, sprintf( ...
            'sawfm_level_%02d__jnr_%ddB', levelIndex, level));
        cfg.experiment.name = 'dual-axis-frequency-development-v2';
        cfg.experiment.power_sweep.fixed_codes = ...
            {'FMB', 'SinFM', 'MFT', 'VFT', 'ISRJ'};
        cfg.experiment.power_sweep.swept_codes = {'SawFM'};
        cfg.experiment.power_sweep.fixed_JNR_dB = 30;
        cfg.experiment.power_sweep.sweep_JNR_dB = level;
        cfg.experiment.power_sweep.standard_levels_dB = standardLevels;
        cfg.experiment.power_sweep.level_index = levelIndex;
        cfg.experiment.power_sweep.paired_seeds = true;
        configs{index} = cfg;
    end

    existingPolicy = 'error';
    result = run_power_sweep_levels(configs, executeGeneration, ...
        'train', existingPolicy, executeGeneration);

    receipt = struct();
    receipt.status = 'completed';
    receipt.executed = executeGeneration;
    receipt.experiment = 'dual-axis-frequency-development-v2';
    receipt.base_seed = baseSeed;
    receipt.levels_dB = levels_dB;
    receipt.high_visibility_JNR_dB = 30;
    receipt.low_visibility_JNR_dB = 20;
    receipt.fixed_other_label_JNR_dB = 30;
    receipt.fixed_codes = {'FMB', 'SinFM', 'MFT', 'VFT', 'ISRJ'};
    receipt.swept_codes = {'SawFM'};
    receipt.split_selection = 'train_only';
    receipt.existing_policy = existingPolicy;
    receipt.sweep_root = sweepRoot;
    receipt.manifest_path = result.manifest_path;
    receipt.paired_seed_verified = result.paired_seed_verified;
    receipt.train_seed_signatures = {result.levels.train_seed_signature};
    receipt.train_statuses = {result.levels.train_status};
    receipt.expected_samples_per_level = 25200;
    receipt.expected_combination_count = 63;
    receipt.samples_per_combination = 400;
    receipt.receiver_identity_contract = ...
        ['same receiver_noise_seed, labels, physical parameters, and ', ...
         'background across 20/30 dB views; paired views share one role'];
    receipt.frequency_manifest_contract = ...
        ['FB and label/combo IF10/20/50 are derived after generation; ', ...
         'paired power views are never counted as independent identities'];
    receipt.test_views_generated = false;
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
        '%s_generation_receipt.json', prefix));
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0
        error('RouteA:FrequencyV2ReceiptWriteFailed', ...
            'Cannot write %s', receiptPath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_FREQUENCY_V2_RECEIPT=%s\n', receiptPath);
end

