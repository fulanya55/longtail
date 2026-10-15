function result = generate_p2e_sawfm_component_oracle_data(executeGeneration)
%GENERATE_P2E_SAWFM_COMPONENT_ORACLE_DATA Fresh 50 dB train-only probe.
    if nargin < 1
        executeGeneration = false;
    end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
        'p2e_sawfm_component_oracle_seed1600000000');
    cfg = create_power_sweep_6class_config(50);
    cfg.seed = 1600000000;
    cfg.dataset.samples_per_pattern = 10;
    cfg.dataset.num_samples = 63 * 10;
    cfg.experiment.name = 'p2e_sawfm_component_oracle_seed1600000000';
    cfg.output_dir = fullfile(outputRoot, 'level_09__jnr_50dB');
    result = run_power_sweep_levels({cfg}, logical(executeGeneration), ...
        'train', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
