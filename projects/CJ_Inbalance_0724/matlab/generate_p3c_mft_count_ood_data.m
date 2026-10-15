function results = generate_p3c_mft_count_ood_data(executeGeneration, baseSeed, outputPrefix)
%GENERATE_P3C_MFT_COUNT_OOD_DATA Paired MFT count and spacing stress regimes.
    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        baseSeed = 450000000;
    end
    if nargin < 3
        outputPrefix = sprintf('p3c_mft_count_ood_seed%d', baseSeed);
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    names = {'ID_control', 'low_count', 'close_spacing', 'ultra_close', 'weak_target'};
    countRanges = {[3, 8], [1, 2], [3, 8], [3, 8], [3, 8]};
    delayRanges = {[30e-6, 50e-6], [30e-6, 50e-6], [30e-6, 34e-6], ...
        [30e-6, 31e-6], [30e-6, 50e-6]};
    amplitudeRanges = {[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], ...
        [0.5, 1.5], [0.05, 1.5]};
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
            cfg.params.MFT.num_targets = countRanges{regimeIndex};
            cfg.params.MFT.target_delay_s = delayRanges{regimeIndex};
            cfg.params.MFT.target_relative_amplitude = amplitudeRanges{regimeIndex};
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
            countRange = countRanges{regimeIndex}; %#ok<NASGU>
            delayRange_s = delayRanges{regimeIndex}; %#ok<NASGU>
            amplitudeRange = amplitudeRanges{regimeIndex}; %#ok<NASGU>
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', ...
                'countRange', 'delayRange_s', 'amplitudeRange', '-v7.3');
        end
    end
end
