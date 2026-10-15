function results = generate_p5f_shared_execution_lazy_head_development_data(executeGeneration)
%GENERATE_P5F_SHARED_EXECUTION_LAZY_HEAD_DEVELOPMENT_DATA Three fresh train-only pools.
    if nargin < 1, executeGeneration = false; end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [970000000, 980000000, 990000000]; levels = 10:5:30; results = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        outputRoot = fullfile(projectRoot, 'generated_data', sprintf('p5f_shared_execution_lazy_head_dev_seed%d', seed));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = seed; cfg.dataset.samples_per_pattern = 60; cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
            cfg.experiment.name = sprintf('p5f_shared_execution_lazy_head_dev_seed%d', seed);
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
            configs{index} = cfg;
        end
        results{seedIndex} = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
        if executeGeneration, save(fullfile(outputRoot, 'generation_receipt.mat'), 'results', '-v7.3'); end
    end
end
