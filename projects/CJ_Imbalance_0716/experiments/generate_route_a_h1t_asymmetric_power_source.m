function receipt = generate_route_a_h1t_asymmetric_power_source(executeGeneration)
%GENERATE_ROUTE_A_H1T_ASYMMETRIC_POWER_SOURCE Fresh train-only H1t power views.
% The three views share every receiver identity, labels and physical random
% substream.  They differ only in the jointly swept MFT/VFT/ISRJ JNR (10, 30,
% or 50 dB); no test split or power-test view is generated.

    if nargin < 1
        executeGeneration = false;
    end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0803_h1t_asymmetric_power_v1';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_h1t_source_generation');
    levels = [10, 30, 50];
    samplesPerPattern = 128;
    baseSeed = 160000000;
    expectedCount = 63 * samplesPerPattern;
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));

    currentRange = [baseSeed + 1, baseSeed + expectedCount];
    reservedRanges = [ ...
        20260732,20285931;23260732,23285931;24000001,24025200;27000001,27025200; ...
        34000001,34025200;37000001,37025200;44000001,44025200;47000001,47025200; ...
        54000001,54025200;57000001,57025200;64000001,64005040;74000001,74025200; ...
        77000001,77025200;84000001,84025200;87000001,87025200;94000001,94025200; ...
        97000001,97025200;104000001,104025200;107000001,107025200;114000001,114025200; ...
        117000001,117025200;124000001,124025200;144000001,144025200;450000001,450025200];
    for index = 1:size(reservedRanges, 1)
        known = reservedRanges(index, :);
        if ~(currentRange(2) < known(1) || currentRange(1) > known(2))
            error('RouteA:IdentityOverlap', 'H1t range overlaps a reserved historical interval.');
        end
    end

    configs = cell(numel(levels), 1);
    standardLevels = 10:5:50;
    for index = 1:numel(levels)
        level = levels(index);
        cfg = create_power_sweep_6class_config(level);
        cfg.seed = baseSeed;
        cfg.dataset.samples_per_pattern = samplesPerPattern;
        cfg.dataset.num_samples = expectedCount;
        levelIndex = find(standardLevels == level, 1, 'first');
        cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, level));
        cfg.experiment.name = 'route_a_h1t_asymmetric_power_v1';
        configs{index} = cfg;
    end
    result = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
    if ~result.paired_seed_verified
        error('RouteA:PairContract', 'Power-view identities were not verified as paired.');
    end

    receiverOffset = mod(2 * 2246822519, 4294967296);
    receiverRange = mod(currentRange + receiverOffset, 4294967296);
    receipt = struct('status', 'completed', 'executed', logical(executeGeneration), ...
        'experiment', 'RouteA_H1t_AsymmetricPowerTransfer_Source', 'evidence_role', 'fresh_identity_train_only_mechanism_source', ...
        'base_seed', baseSeed, 'samples_per_pattern', samplesPerPattern, 'expected_samples_per_level', expectedCount, ...
        'levels_dB', levels, 'fixed_jnr_dB', 30, 'swept_codes', {{'MFT','VFT','ISRJ'}}, ...
        'output_root', outputRoot, 'manifest_path', result.manifest_path, 'train_only', true, ...
        'current_sample_seed_range', currentRange, 'expected_receiver_noise_seed_range', receiverRange, ...
        'receiver_noise_mapping', 'mod(sample_seed + 2*2246822519, 2^32)', ...
        'identity_disjoint_precheck', true, ...
        'paired_seed_verified', result.paired_seed_verified, 'test_split_generated', false, ...
        'locked_test_used', false, 'test_v2_used', false, 'test_v3_used', false);
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix = 'dry_run'; if executeGeneration; prefix = 'formal'; end
    receiptPath = fullfile(auditRoot, sprintf('%s_h1t_source_generation_receipt.json', prefix));
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0; error('RouteA:ReceiptWriteFailed', 'Cannot write receipt.'); end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_H1T_SOURCE_RECEIPT=%s\n', receiptPath);
end
