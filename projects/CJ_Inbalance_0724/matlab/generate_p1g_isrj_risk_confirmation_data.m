function result = generate_p1g_isrj_risk_confirmation_data(executeGeneration)
%GENERATE_P1G_ISRJ_RISK_CONFIRMATION_DATA Fresh confirmation for P1g.

    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    levels = [20, 25];
    allLevels = 10:5:50;
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', ...
        'generated_data', 'p1g_isrj_risk_confirmation_seed400000000');
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_power_sweep_6class_config(levels(index));
        cfg.seed = 400000000;
        cfg.dataset.samples_per_pattern = 40;
        cfg.dataset.num_samples = 63 * 40;
        cfg.experiment.name = 'p1g_isrj_risk_confirmation_seed400000000';
        levelIndex = find(allLevels == levels(index), 1, 'first');
        levelDirectory = sprintf('level_%02d__jnr_%ddB', levelIndex, levels(index));
        cfg.output_dir = fullfile(outputRoot, levelDirectory);
        configs{index} = cfg;
    end
    % Resume validates already committed MAT files before reusing them.  This
    % makes an interrupted multi-level generation recoverable without any
    % overwrite of completed evidence.
    result = run_power_sweep_levels(configs, logical(executeGeneration), ...
        'both', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
