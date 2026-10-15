function result = generate_p1k_joint_level3_data(executeGeneration)
%GENERATE_P1K_JOINT_LEVEL3_DATA Fresh identity-safe data for frozen P1k.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    levels = [20, 25]; allLevels = 10:5:50;
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
        'p1k_joint_level3_seed800000000');
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_power_sweep_6class_config(levels(index));
        cfg.seed = 800000000;
        cfg.dataset.samples_per_pattern = 40;
        cfg.dataset.num_samples = 63 * 40;
        cfg.experiment.name = 'p1k_joint_level3_seed800000000';
        levelIndex = find(allLevels == levels(index), 1, 'first');
        cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, levels(index)));
        configs{index} = cfg;
    end
    result = run_power_sweep_levels(configs, logical(executeGeneration), ...
        'both', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
