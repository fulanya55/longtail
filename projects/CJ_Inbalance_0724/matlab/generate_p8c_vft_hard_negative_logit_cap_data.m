function result = generate_p8c_vft_hard_negative_logit_cap_data(executeGeneration)
%GENERATE_P8C_VFT_HARD_NEGATIVE_LOGIT_CAP_DATA Fresh paired nine-power train-only data.
    if nargin < 1, executeGeneration = false; end
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    levels = 10:5:50;
    outputRoot = fullfile(projectRoot, 'generated_data', 'p8c_vft_hard_negative_logit_cap_seed1325000000');
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_power_sweep_6class_config(levels(index));
        cfg.seed = 1325000000;
        cfg.dataset.samples_per_pattern = 400;
        cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
        cfg.experiment.name = 'p8c_vft_hard_negative_logit_cap_seed1325000000';
        cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
        configs{index} = cfg;
    end
    result = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
