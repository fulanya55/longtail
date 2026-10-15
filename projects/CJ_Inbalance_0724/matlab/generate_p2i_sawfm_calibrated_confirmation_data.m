function results = generate_p2i_sawfm_calibrated_confirmation_data(executeGeneration)
%GENERATE_P2I_SAWFM_CALIBRATED_CONFIRMATION_DATA Two fresh train-only seeds.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [2100000000, 2200000000];
    levels = [35, 40, 45, 50];
    allLevels = 10:5:50;
    results = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
            sprintf('p2i_sawfm_calibrated_confirmation_seed%d', seed));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = seed;
            cfg.dataset.samples_per_pattern = 10;
            cfg.dataset.num_samples = 63 * 10;
            cfg.experiment.name = sprintf('p2i_sawfm_calibrated_confirmation_seed%d', seed);
            levelIndex = find(allLevels == levels(index), 1, 'first');
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, levels(index)));
            configs{index} = cfg;
        end
        results{seedIndex} = run_power_sweep_levels(configs, logical(executeGeneration), ...
            'train', 'resume', logical(executeGeneration));
        if executeGeneration
            result = results{seedIndex}; %#ok<NASGU>
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
        end
    end
end
