function result = generate_p4c_heterogeneous_route_variable_jnr_data(executeGeneration)
%GENERATE_P4C_HETEROGENEOUS_ROUTE_VARIABLE_JNR_DATA
% Generate a train-only ordinary closed-set pool with JNR sampled in 10--30 dB.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));

    cfg = create_closedset_6class_config();
    cfg.seed = 590000000;
    cfg.dataset.samples_per_pattern = 100;
    cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
    cfg.params.common.JNR_dB = [10, 30];
    cfg.experiment.name = 'p4c_heterogeneous_route_variable_jnr_seed590000000';
    cfg.output_dir = fullfile(projectRoot, 'generated_data', ...
        'p4c_heterogeneous_route_variable_jnr_seed590000000');
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), ...
        'train', 'resume');
    if executeGeneration
        save(fullfile(cfg.output_dir, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
