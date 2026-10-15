function results = generate_p5q_relocation_safe_checkpoint_development_data(executeGeneration)
%GENERATE_P5Q_RELOCATION_SAFE_CHECKPOINT_DEVELOPMENT_DATA Fresh P5q pools.
% Resolve both the project and generator from this relocated file. This
% function creates train-only paired-power views and never creates test data.
    if nargin < 1, executeGeneration = false; end
    thisFile = mfilename('fullpath');
    projectRoot = fileparts(fileparts(thisFile));
    ljqRoot = fileparts(projectRoot);
    generatorRoot = fullfile(ljqRoot, '干扰数据', '干扰信号生成程序', '可配置组合干扰数据生成');
    if ~isfolder(generatorRoot)
        error('P5q generator root is missing after relocation: %s', generatorRoot);
    end
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [3710000000, 3720000000, 3730000000];
    levels = 10:5:30;
    results = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        outputRoot = fullfile(projectRoot, 'generated_data', sprintf('p5q_relocation_safe_checkpoint_dev_seed%d', seed));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = seed;
            cfg.dataset.samples_per_pattern = 60;
            cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
            cfg.experiment.name = sprintf('p5q_relocation_safe_checkpoint_dev_seed%d', seed);
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
            configs{index} = cfg;
        end
        results{seedIndex} = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
        if executeGeneration
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'results', '-v7.3');
        end
    end
end
