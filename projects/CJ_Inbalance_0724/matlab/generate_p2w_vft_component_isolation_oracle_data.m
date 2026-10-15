function result = generate_p2w_vft_component_isolation_oracle_data(executeGeneration)
%GENERATE_P2W_VFT_COMPONENT_ISOLATION_ORACLE_DATA Fresh train-only oracle pools.
    if nargin < 1, executeGeneration = false; end
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    seeds = [1381000000, 1382000000, 1383000000];
    powers = [10, 30, 50];
    result = cell(numel(seeds), 1);
    for seedIndex = 1:numel(seeds)
        seed = seeds(seedIndex);
        outputRoot = fullfile(projectRoot, 'generated_data', sprintf('p2w_vft_component_oracle_seed%d', seed));
        configs = cell(numel(powers), 1);
        for powerIndex = 1:numel(powers)
            cfg = create_power_sweep_6class_config(powers(powerIndex));
            cfg.seed = seed;
            cfg.dataset.samples_per_pattern = 10;
            cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
            cfg.experiment.name = sprintf('p2w_vft_component_oracle_seed%d', seed);
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', powerIndex, powers(powerIndex)));
            configs{powerIndex} = cfg;
        end
        result{seedIndex} = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
        if executeGeneration
            for powerIndex = 1:numel(powers)
                write_vft_oracle_component(fullfile(configs{powerIndex}.output_dir, 'closedset_6class_train.mat'), configs{powerIndex});
            end
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', '-v7.3');
        end
    end
end

function write_vft_oracle_component(dataPath, cfg)
% Persist clean, calibrated VFT only for VFT-positive identities.  This is oracle-only.
    cfg = normalize_and_validate_config(cfg, 'balanced');
    source = load(dataPath, 'label_selected', 'component_seeds', 'component_target_power_dB');
    % MATLAB loads the v7.3 sample-first label matrix as [N, 6].
    labels = logical(source.label_selected);
    vftIndex = 5;
    selected = find(labels(:, vftIndex));
    registry = build_jamming_registry(cfg);
    reference = build_reference_signal(cfg);
    vft_oracle_iq = zeros(6000, numel(selected), 'single'); %#ok<NASGU>
    vft_oracle_indices = uint32(selected); %#ok<NASGU>
    for outputIndex = 1:numel(selected)
        sampleIndex = selected(outputIndex);
        componentIndex = sum(labels(sampleIndex, 1:vftIndex));
        if componentIndex < 1 || componentIndex > numel(source.component_seeds{sampleIndex}) || componentIndex > numel(source.component_target_power_dB{sampleIndex})
            error('P2w:OracleComponentContract', 'VFT component order or seed/power cardinality is invalid.');
        end
        seed = source.component_seeds{sampleIndex}(componentIndex);
        targetPower = source.component_target_power_dB{sampleIndex}(componentIndex);
        stream = RandStream('mt19937ar', 'Seed', seed);
        raw = generate_single_jamming('VFT', cfg, registry, stream, reference);
        scaled = calibrate_component_power(raw, targetPower);
        vft_oracle_iq(:, outputIndex) = single([real(scaled(:)); imag(scaled(:))]);
    end
    outputPath = fullfile(fileparts(dataPath), 'vft_oracle_component.mat');
    save(outputPath, 'vft_oracle_iq', 'vft_oracle_indices', '-v7.3');
end
