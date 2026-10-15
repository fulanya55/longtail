function result = generate_p1d_independent_power_sweep(executeGeneration)
%GENERATE_P1D_INDEPENDENT_POWER_SWEEP Generate a small independent 9-level set.
% Only the base seed, samples per pattern, experiment name, and output root
% differ from the audited 0716 power-sweep protocol.

    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    levels = 10:5:50;
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', ...
        'generated_data', 'p1d_independent_power_sweep_seed100000000');
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_power_sweep_6class_config(levels(index));
        cfg.seed = 100000000;
        cfg.dataset.samples_per_pattern = 20;
        cfg.dataset.num_samples = 63 * 20;
        cfg.experiment.name = 'p1d_independent_power_sweep_seed100000000';
        levelDirectory = sprintf('level_%02d__jnr_%ddB', index, levels(index));
        cfg.output_dir = fullfile(outputRoot, levelDirectory);
        configs{index} = cfg;
    end
    result = run_power_sweep_levels(configs, logical(executeGeneration), ...
        'both', 'error', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
