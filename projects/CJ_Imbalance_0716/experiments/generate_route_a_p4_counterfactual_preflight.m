function outputPath = generate_route_a_p4_counterfactual_preflight(outputDir, samplesPerPair)
% Generate a tiny, isolated counterfactual pair preflight for Route-A P4.
% Each pair shares sample_seed (therefore receiver noise/reference stream)
% while changing only the selected six-class combination. Original MAT files
% are never opened or modified.
    if nargin < 1
        outputDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'outputs', 'route_a_p4_counterfactual_preflight_20260831');
    end
    if nargin < 2, samplesPerPair = 1; end
    validateattributes(samplesPerPair, {'numeric'}, {'scalar','integer','positive'});
    root = fileparts(mfilename('fullpath'));
    generatorRoot = fullfile('E:', 'LJQ_code', '干扰数据', ...
        '干扰信号生成程序', '可配置组合干扰数据生成');
    addpath(genpath(generatorRoot));
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    cfg = create_default_config();
    selected = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    % Use only the target six classes for this isolated preflight.  This is a
    % derived fixture and does not modify the nine-class source data.
    cfg.types.enabled = selected;
    cfg.types.known = selected;
    cfg.types.unknown = {};
    cfg.radar.num_pulses = 1;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 6;
    cfg.dataset.samples_per_pattern = 1;
    cfg.dataset.num_samples = 63;
    cfg.params.common.JNR_dB = [30, 30];
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_balanced_manifest(cfg, registry, 'train');
    reference = build_reference_signal(cfg);

    pairs = {
        'FMB', 'AMA'; 'FMB', 'SawFM'; 'FMB', 'SinFM';
        'FMB', 'MFT'; 'FMB', 'ISRJ'; 'AMA', 'SawFM';
        'AMA', 'SinFM'; 'AMA', 'MFT'; 'AMA', 'ISRJ';
        'SawFM', 'SinFM'; 'SawFM', 'MFT'; 'SawFM', 'ISRJ';
        'SinFM', 'MFT'; 'SinFM', 'ISRJ'; 'MFT', 'ISRJ';
        'FMB+AMA', 'FMB+SawFM'; 'FMB+SinFM', 'FMB+MFT';
        'FMB+ISRJ', 'AMA+ISRJ'; 'SawFM+MFT', 'SawFM+SinFM';
        'SinFM+ISRJ', 'MFT+ISRJ'};
    pairCount = size(pairs, 1);
    totalCount = pairCount * samplesPerPair;
    sampleLength = cfg.derived.sample_length;
    iq_a = complex(zeros(sampleLength, totalCount));
    iq_b = complex(zeros(sampleLength, totalCount));
    labels_a = false(totalCount, numel(selected));
    labels_b = false(totalCount, numel(selected));
    shared_sample_seed = zeros(totalCount, 1, 'uint32');
    pair_group_id = zeros(totalCount, 1, 'uint32');
    pair_variant = zeros(totalCount, 1, 'uint8');
    pair_codes = cell(pairCount, 2);

    for pairIndex = 1:pairCount
        codeA = pairs{pairIndex, 1}; codeB = pairs{pairIndex, 2};
        rowA = manifest(strcmp(manifest.combination_id, codeA), :);
        rowB = manifest(strcmp(manifest.combination_id, codeB), :);
        if height(rowA) ~= 1 || height(rowB) ~= 1
            error('RouteA:P4PairMissing', 'Pair combination missing: %s / %s', codeA, codeB);
        end
        for repeatIndex = 1:samplesPerPair
            outputIndex = (pairIndex - 1) * samplesPerPair + repeatIndex;
            sharedSeed = uint32(31000000 + outputIndex);
            rowA.sample_id = 100000 + outputIndex * 2 - 1;
            rowB.sample_id = 100000 + outputIndex * 2;
            rowA.sample_seed = sharedSeed;
            rowB.sample_seed = sharedSeed;
            sampleA = compose_jamming_sample(cfg, registry, schema, rowA, reference);
            sampleB = compose_jamming_sample(cfg, registry, schema, rowB, reference);
            iq_a(:, outputIndex) = sampleA.iq_complex(:);
            iq_b(:, outputIndex) = sampleB.iq_complex(:);
            registryCodes = {registry.code};
            labels_a(outputIndex, :) = logical(sampleA.label_all(ismember(registryCodes, selected)));
            labels_b(outputIndex, :) = logical(sampleB.label_all(ismember(registryCodes, selected)));
            shared_sample_seed(outputIndex) = sharedSeed;
            pair_group_id(outputIndex) = pairIndex;
            pair_variant(outputIndex) = 0;
        end
        pair_codes(pairIndex, :) = {codeA, codeB};
    end

    outputPath = fullfile(outputDir, 'route_a_p4_counterfactual_preflight.mat');
    save(outputPath, 'iq_a', 'iq_b', 'labels_a', 'labels_b', ...
        'shared_sample_seed', 'pair_group_id', 'pair_variant', ...
        'pair_codes', 'selected', 'samplesPerPair', '-v7.3');
    receipt = struct();
    receipt.status = 'completed';
    receipt.pair_count = pairCount;
    receipt.samples_per_pair = samplesPerPair;
    receipt.total_pair_count = totalCount;
    receipt.shared_seed_unique_count = numel(unique(shared_sample_seed));
    receipt.same_noise_seed_within_pair = true;
    receipt.original_data_opened = false;
    receipt.output_path = outputPath;
    save(fullfile(outputDir, 'generation_receipt.mat'), 'receipt', '-v7.3');
end
