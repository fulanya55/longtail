function results = generate_p5n_verified_idle_window_shared_dense_development_data(executeGeneration)
%GENERATE_P5N_VERIFIED_IDLE_WINDOW_SHARED_DENSE_DEVELOPMENT_DATA Fresh P5n pools.
% This successor changes only quarantined P5m identities. It creates five
% paired train-only power views and never creates a validation or test split.
    if nargin < 1, executeGeneration = false; end
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    % Recover the validated non-ASCII generator path from P5m rather than
    % relying on a duplicated path literal in this successor wrapper.
    p5mSource = which('generate_p5m_authorized_window_shared_dense_development_data');
    if isempty(p5mSource), error('P5n requires the project matlab folder on path.'); end
    p5mText = fileread(p5mSource);
    match = regexp(p5mText, "generatorRoot = '([^']+)';", 'tokens', 'once');
    if isempty(match), error('P5n could not recover the validated P5m generator root.'); end
    generatorRoot = match{1};
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [4140000000, 4160000000, 4170000000];
    levels = 10:5:30;
    results = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        outputRoot = fullfile(projectRoot, 'generated_data', sprintf('p5n_verified_idle_window_dev_seed%d', seed));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = seed;
            cfg.dataset.samples_per_pattern = 60;
            cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
            cfg.experiment.name = sprintf('p5n_verified_idle_window_dev_seed%d', seed);
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
            configs{index} = cfg;
        end
        results{seedIndex} = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
        if executeGeneration
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'results', '-v7.3');
        end
    end
end
