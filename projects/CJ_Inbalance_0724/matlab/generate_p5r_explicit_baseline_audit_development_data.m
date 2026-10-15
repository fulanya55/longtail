function results = generate_p5r_explicit_baseline_audit_development_data(executeGeneration)
%GENERATE_P5R_EXPLICIT_BASELINE_AUDIT_DEVELOPMENT_DATA Fresh P5r pools.
    if nargin < 1, executeGeneration = false; end
    thisFile = mfilename('fullpath');
    projectRoot = fileparts(fileparts(thisFile));
    ljqRoot = fileparts(projectRoot);
    generatorRoot = fullfile(ljqRoot, '干扰数据', '干扰信号生成程序', '可配置组合干扰数据生成');
    if ~isfolder(generatorRoot), error('P5r generator root is missing: %s', generatorRoot); end
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [3610000000, 3620000000, 3630000000]; levels = 10:5:30; results = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex); outputRoot = fullfile(projectRoot, 'generated_data', sprintf('p5r_explicit_baseline_audit_dev_seed%d', seed)); configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index)); cfg.seed = seed; cfg.dataset.samples_per_pattern = 60; cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
            cfg.experiment.name = sprintf('p5r_explicit_baseline_audit_dev_seed%d', seed); cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index))); configs{index} = cfg;
        end
        results{seedIndex} = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
        if executeGeneration, save(fullfile(outputRoot, 'generation_receipt.mat'), 'results', '-v7.3'); end
    end
end
