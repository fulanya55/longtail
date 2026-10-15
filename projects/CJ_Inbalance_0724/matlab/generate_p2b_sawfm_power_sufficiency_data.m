function result = generate_p2b_sawfm_power_sufficiency_data(executeGeneration)
%GENERATE_P2B_SAWFM_POWER_SUFFICIENCY_DATA Fresh train-only development-v2.
% The test split is intentionally not generated before development promotion.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    levels = 10:5:50;
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
        'p2b_sawfm_power_sufficiency_seed1300000000');
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_power_sweep_6class_config(levels(index));
        cfg.seed = 1300000000;
        cfg.dataset.samples_per_pattern = 40;
        cfg.dataset.num_samples = 63 * 40;
        cfg.experiment.name = 'p2b_sawfm_power_sufficiency_seed1300000000';
        cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
        configs{index} = cfg;
    end
    result = run_power_sweep_levels(configs, logical(executeGeneration), ...
        'train', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
