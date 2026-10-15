function result = generate_route_a_locked_test_source(executeGeneration)
%GENERATE_ROUTE_A_LOCKED_TEST_SOURCE
% Create a fresh E-drive, test-only nine-power source for Route-A's new
% six-class branch.  The caller must explicitly pass true to write data.
% This function never reads an existing test file and refuses to overwrite
% an existing output directory/file.

    if nargin < 1
        executeGeneration = false;
    end
    if ~(islogical(executeGeneration) || (isnumeric(executeGeneration) && isscalar(executeGeneration)))
        error('RouteA:InvalidExecutionFlag', 'executeGeneration must be logical or 0/1.');
    end
    executeGeneration = logical(executeGeneration);

    generatorRoot = fullfile('E:', 'LJQ_code', '干扰数据', '干扰信号生成程序', '可配置组合干扰数据生成');
    if ~isfolder(generatorRoot)
        error('RouteA:GeneratorRootMissing', 'Generator root missing: %s', generatorRoot);
    end
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));

    levels = 10:5:50;
    outputRoot = fullfile('E:', 'LJQ_code', 'CJ_Imbalance_0716', 'generated_data', ...
        'route_a_locked_test_ama_seed4050000000');
    if executeGeneration && isfolder(outputRoot)
        error('RouteA:OutputExists', 'Refusing to overwrite existing output root: %s', outputRoot);
    end

    selectedTypes = {'FMB', 'AMA', 'SawFM', 'SinFM', 'MFT', 'ISRJ'};
    configs = cell(numel(levels), 1);
    for index = 1:numel(levels)
        cfg = create_closedset_6class_config();
        cfg.seed = 4050000000;
        cfg.types.enabled = selectedTypes;
        cfg.types.known = selectedTypes;
        cfg.types.unknown = {};
        cfg.dataset.samples_per_pattern = 40;
        cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
        cfg.output.save_v73 = true;
        cfg.output.save_legacy = false;
        cfg.params.common.JNR_dB = [30, 30];
        cfg.experiment.name = 'route_a_locked_test_ama_seed4050000000';
        cfg.experiment.power_sweep.fixed_codes = {'FMB', 'SawFM', 'SinFM'};
        cfg.experiment.power_sweep.swept_codes = {'AMA', 'MFT', 'ISRJ'};
        cfg.experiment.power_sweep.fixed_JNR_dB = 30;
        cfg.experiment.power_sweep.sweep_JNR_dB = levels(index);
        cfg.experiment.power_sweep.standard_levels_dB = levels;
        cfg.experiment.power_sweep.level_index = index;
        cfg.experiment.power_sweep.paired_seeds = true;
        cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', index, levels(index)));
        configs{index} = cfg;
    end

    % Test-only generation keeps this source out of all development roles.
    result = run_power_sweep_levels(configs, executeGeneration, ...
        'test', 'error', executeGeneration);
    result.route_a_contract = struct( ...
        'selected_classes', {selectedTypes}, ...
        'excluded_classes', {{'SN', 'VFT', 'RangeFakeTarget'}}, ...
        'generation_seed', 4050000000, ...
        'samples_per_pattern', 40, ...
        'split_selection', 'test', ...
        'locked_test_used', false, ...
        'test_files_opened', false);
end
