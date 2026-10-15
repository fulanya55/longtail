function result = generate_p2b_nine_power_confirmation_data(executeGeneration)
%GENERATE_P2B_NINE_POWER_CONFIRMATION_DATA Fresh confirmation for P2b.
% The P2a helper is invoked in dry-run mode only to register the external
% generator paths without duplicating machine-specific non-ASCII paths here.
    if nargin < 1
        executeGeneration = false;
    end
    generate_p2a_nine_power_generalization_data(false);
    levels = 10:5:50;
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
        'p2b_nine_power_confirmation_seed1300000000');
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_power_sweep_6class_config(levels(index));
        cfg.seed = 1300000000;
        cfg.dataset.samples_per_pattern = 40;
        cfg.dataset.num_samples = 63 * 40;
        cfg.experiment.name = 'p2b_nine_power_confirmation_seed1300000000';
        cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
        configs{index} = cfg;
    end
    result = run_power_sweep_levels(configs, logical(executeGeneration), ...
        'both', 'resume', logical(executeGeneration));
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
    end
end
