function results = generate_p2m_analytic_saw_period_data(executeGeneration, baseSeed, outputPrefix)
%GENERATE_P2M_ANALYTIC_SAW_PERIOD_DATA Fresh paired OOD data for FFT-ACF screen.
    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        baseSeed = 2700000000;
    end
    if nargin < 3
        outputPrefix = sprintf('p2m_analytic_saw_period_seed%d', baseSeed);
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    names = {'ID_control', 'near_low', 'near_high', 'far_low', 'far_high'};
    ranges = {[25e-6, 75e-6], [10e-6, 20e-6], [80e-6, 100e-6], [5e-6, 10e-6], [100e-6, 160e-6]};
    levels = [25, 35, 50];
    allLevels = 10:5:50;
    results = cell(numel(names), 1);
    for regimeIndex = 1:numel(names)
        name = names{regimeIndex};
        outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
            sprintf('%s_%s', outputPrefix, name));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = baseSeed;
            cfg.params.SawFM.sweep_period_s = ranges{regimeIndex};
            cfg.dataset.samples_per_pattern = 10;
            cfg.dataset.num_samples = 63 * 10;
            cfg.experiment.name = sprintf('%s_%s', outputPrefix, name);
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
