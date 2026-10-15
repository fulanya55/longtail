function results = generate_p2o_analytic_ood_confirmation_data(executeGeneration)
%GENERATE_P2O_ANALYTIC_OOD_CONFIRMATION_DATA Two fresh five-regime data seeds.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [2800000000, 2900000000];
    names = {'ID_control', 'near_low', 'near_high', 'far_low', 'far_high'};
    ranges = {[25e-6, 75e-6], [10e-6, 20e-6], [80e-6, 100e-6], [5e-6, 10e-6], [100e-6, 160e-6]};
    levels = [25, 35, 50];
    allLevels = 10:5:50;
    results = cell(numel(seeds), numel(names));
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        for regimeIndex = 1:numel(names)
            name = names{regimeIndex};
            outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
                sprintf('p2o_analytic_ood_confirmation_seed%d_%s', seed, name));
            configs = cell(numel(levels), 1);
            for index = 1:numel(levels)
                cfg = create_power_sweep_6class_config(levels(index));
                cfg.seed = seed;
                cfg.params.SawFM.sweep_period_s = ranges{regimeIndex};
                cfg.dataset.samples_per_pattern = 10;
                cfg.dataset.num_samples = 63 * 10;
                cfg.experiment.name = sprintf('p2o_analytic_ood_confirmation_seed%d_%s', seed, name);
                levelIndex = find(allLevels == levels(index), 1, 'first');
                cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, levels(index)));
                configs{index} = cfg;
            end
            results{seedIndex, regimeIndex} = run_power_sweep_levels(configs, logical(executeGeneration), ...
                'train', 'resume', logical(executeGeneration));
            if executeGeneration
                result = results{seedIndex, regimeIndex}; %#ok<NASGU>
                periodRange_s = ranges{regimeIndex}; %#ok<NASGU>
                save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', 'periodRange_s', '-v7.3');
            end
        end
    end
end
