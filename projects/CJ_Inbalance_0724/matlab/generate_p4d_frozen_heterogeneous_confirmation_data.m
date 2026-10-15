function results = generate_p4d_frozen_heterogeneous_confirmation_data(executeGeneration)
%GENERATE_P4D_FROZEN_HETEROGENEOUS_CONFIRMATION_DATA
% Generate two train-container-only pools for frozen, no-tuning evaluation.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));

    seeds = [670000000, 680000000];
    results = cell(size(seeds));
    for index = 1:numel(seeds)
        dataSeed = seeds(index);
        cfg = create_closedset_6class_config();
        cfg.seed = dataSeed;
        cfg.dataset.samples_per_pattern = 100;
        cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
        cfg.params.common.JNR_dB = [10, 30];
        cfg.experiment.name = sprintf('p4d_frozen_heterogeneous_confirmation_seed%d', dataSeed);
        cfg.output_dir = fullfile(projectRoot, 'generated_data', cfg.experiment.name);
        cfg = normalize_and_validate_config(cfg, 'balanced');
        registry = build_jamming_registry(cfg);
        schema = build_parameter_schema(registry);
        results{index} = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), ...
            'train', 'resume');
        if executeGeneration
            result = results{index}; %#ok<NASGU>
            save(fullfile(cfg.output_dir, 'generation_receipt.mat'), 'result', '-v7.3');
        end
    end
end
