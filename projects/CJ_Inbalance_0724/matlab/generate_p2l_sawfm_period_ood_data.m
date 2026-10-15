function results = generate_p2l_sawfm_period_ood_data(executeGeneration)
%GENERATE_P2L_SAWFM_PERIOD_OOD_DATA Paired ID/near/far SawFM period regimes.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    names = {'ID_control', 'near_low', 'near_high', 'far_low', 'far_high'};
    ranges = {
        [25e-6, 75e-6], ...
        [10e-6, 20e-6], ...
        [80e-6, 100e-6], ...
        [5e-6, 10e-6], ...
        [100e-6, 160e-6]};
    levels = [25, 35, 50];
    allLevels = 10:5:50;
    results = cell(numel(names), 1);
    for regimeIndex = 1:numel(names)
        name = names{regimeIndex};
        outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
            sprintf('p2l_sawfm_period_ood_seed2600000000_%s', name));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = 2600000000;
            cfg.params.SawFM.sweep_period_s = ranges{regimeIndex};
            cfg.dataset.samples_per_pattern = 10;
            cfg.dataset.num_samples = 63 * 10;
            cfg.experiment.name = sprintf('p2l_sawfm_period_ood_seed2600000000_%s', name);
            levelIndex = find(allLevels == levels(index), 1, 'first');
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, levels(index)));
            configs{index} = cfg;
        end
        results{regimeIndex} = run_power_sweep_levels(configs, logical(executeGeneration), ...
            'train', 'resume', logical(executeGeneration));
        if executeGeneration
            result = results{regimeIndex}; %#ok<NASGU>
            periodRange_s = ranges{regimeIndex}; %#ok<NASGU>
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', 'periodRange_s', '-v7.3');
        end
    end
end
