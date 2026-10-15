function result = generate_p1h_vft_doppler_confirmation_data(executeGeneration)
%GENERATE_P1H_VFT_DOPPLER_CONFIRMATION_DATA Three fresh independent data seeds.

    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));

    seeds = [500000000, 510000000, 520000000];
    levels = [20, 25];
    allLevels = 10:5:50;
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    runs = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        outputRoot = fullfile(projectRoot, 'generated_data', ...
            sprintf('p1h_vft_doppler_seed%d', seed));
        configs = cell(numel(levels), 1);
        for levelIndexLocal = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(levelIndexLocal));
            cfg.seed = seed;
            cfg.dataset.samples_per_pattern = 20;
            cfg.dataset.num_samples = 63 * 20;
            cfg.experiment.name = sprintf('p1h_vft_doppler_seed%d', seed);
            levelIndex = find(allLevels == levels(levelIndexLocal), 1, 'first');
            levelDirectory = sprintf('level_%02d__jnr_%ddB', ...
                levelIndex, levels(levelIndexLocal));
            cfg.output_dir = fullfile(outputRoot, levelDirectory);
            configs{levelIndexLocal} = cfg;
        end
        runs{seedIndex} = run_power_sweep_levels(configs, ...
            logical(executeGeneration), 'both', 'resume', logical(executeGeneration));
        if executeGeneration
            receipt = runs{seedIndex}; %#ok<NASGU>
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'receipt', '-v7.3');
        end
    end
    result = struct('executed', logical(executeGeneration), ...
        'seeds', seeds, 'levels_dB', levels, 'runs', {runs});
    if executeGeneration
        receiptRoot = fullfile(projectRoot, 'generated_data', ...
            'p1h_vft_doppler_confirmation');
        if ~isfolder(receiptRoot)
            mkdir(receiptRoot);
        end
        save(fullfile(receiptRoot, 'combined_generation_receipt.mat'), ...
            'result', '-v7.3');
    end
end
