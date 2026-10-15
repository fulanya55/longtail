function result = generate_p4b_heterogeneous_route_fresh_identity_data(executeGeneration)
%GENERATE_P4B_HETEROGENEOUS_ROUTE_FRESH_IDENTITY_DATA
% Generate one train-only, 30 dB, all-63-combination identity pool.
% No test split is requested or generated.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));

    seed = 510000000;
    powerDb = 30;
    outputRoot = fullfile(projectRoot, 'generated_data', ...
        sprintf('p4b_heterogeneous_route_seed%d', seed));
    cfg = create_power_sweep_6class_config(powerDb);
    cfg.seed = seed;
    cfg.dataset.samples_per_pattern = 100;
    cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
    cfg.experiment.name = sprintf('p4b_heterogeneous_route_seed%d', seed);
    cfg.output_dir = fullfile(outputRoot, 'level_05__jnr_30dB');

    result = run_power_sweep_levels({cfg}, logical(executeGeneration), ...
        'train', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
