function receipts = generate_p2e_stage_b_weak_components(executeGeneration, baseSeed, samplesPerPattern, outputName)
%GENERATE_P2E_STAGE_B_WEAK_COMPONENTS Target-specific -5/-10/-15 dB tests.
% All nine conditions share the same base seed so receiver_noise_seed forms
% controlled counterfactual identity groups across target and weakness views.
    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        baseSeed = 1500000000;
    end
    if nargin < 3
        samplesPerPattern = 40;
    end
    if nargin < 4
        outputName = sprintf('p2e_stage_b_weak_components_seed%d', baseSeed);
    end
    generate_p2a_nine_power_generalization_data(false);
    targets = {'SawFM', 'VFT', 'ISRJ'};
    deficits = [5, 10, 15];
    outputRoot = fullfile('C:\LJQ_code\CJ_Inbalance_0724', 'generated_data', ...
        outputName);
    if executeGeneration && ~isfolder(outputRoot)
        mkdir(outputRoot);
    end
    receipts = repmat(struct(), numel(targets) * numel(deficits), 1);
    row = 0;
    for targetIndex = 1:numel(targets)
        for deficitIndex = 1:numel(deficits)
            row = row + 1;
            target = targets{targetIndex};
            deficit = deficits(deficitIndex);
            cfg = create_closedset_6class_config();
            cfg.seed = baseSeed;
            cfg.dataset.samples_per_pattern = samplesPerPattern;
            cfg.dataset.num_samples = 63 * samplesPerPattern;
            cfg.params.common.JNR_dB = [30, 30];
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.weak_component_policy = 'fixed';
            cfg.imbalance.power.fixed_component = target;
            cfg.imbalance.power.relative_power_dB = [-deficit, -deficit];
            conditionName = sprintf('%s_weak_%02ddB', lower(target), deficit);
            cfg.experiment.name = ['p2e_stage_b_' conditionName];
            cfg.output_dir = fullfile(outputRoot, conditionName);
            if executeGeneration
                dataPath = fullfile(cfg.output_dir, 'compound_dataset_imbalanced.mat');
                if isfile(dataPath)
                    error('P2E:ExistingOutput', 'Refusing to overwrite %s', dataPath);
                end
                generated = generate_imbalanced_compound_dataset(cfg);
                manifest = generated.manifest;
            else
                normalized = normalize_and_validate_config(cfg, 'imbalanced');
                registry = build_jamming_registry(normalized);
                manifest = build_imbalanced_manifest(normalized, registry, 'train');
                dataPath = fullfile(cfg.output_dir, 'compound_dataset_imbalanced.mat');
            end
            targetMask = cellfun(@(codes) any(strcmp(codes, target)), manifest.active_type_codes);
            targetOffset = cellfun(@(codes, offsets) target_power(codes, offsets, target), ...
                manifest.active_type_codes(targetMask), manifest.component_target_power_dB(targetMask));
            receipts(row).condition = conditionName;
            receipts(row).target = target;
            receipts(row).deficit_dB = deficit;
            receipts(row).seed = cfg.seed;
            receipts(row).samples = height(manifest);
            receipts(row).target_positive_samples = sum(targetMask);
            receipts(row).combination_sizes = unique(manifest.combination_size(targetMask))';
            receipts(row).target_offsets_unique_dB = unique(targetOffset)';
            receipts(row).data_path = dataPath;
        end
    end
    if executeGeneration
        save(fullfile(outputRoot, 'generation_receipt.mat'), 'receipts', '-v7.3');
    end
end

function value = target_power(codes, offsets, target)
    index = find(strcmp(codes, target), 1, 'first');
    value = offsets(index);
end
