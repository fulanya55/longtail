function result = generate_route_a_backbone_development_v1(executeGeneration)
%GENERATE_ROUTE_A_BACKBONE_DEVELOPMENT_V1 Generate one train-only 0716 replica.
%   This entrypoint is deliberately separate from the historical closed-set
%   train/test generator.  It creates only the pre-registered Route-A
%   backbone development source and never constructs a test manifest.

    if nargin < 1
        executeGeneration = true;
    end
    if ~(islogical(executeGeneration) && isscalar(executeGeneration))
        error('RouteABackbone:InvalidExecuteFlag', ...
            'executeGeneration must be a logical scalar.');
    end

    projectRoot = 'E:\LJQ_code\CJ_Imbalance_0716';
    generatorRoot = ['E:\LJQ_code\干扰数据\干扰信号生成程序\' ...
        '可配置组合干扰数据生成'];
    outputDir = ['E:\LJQ_code\干扰数据\' ...
        'route_a_backbone-development-v1_20260806'];
    receiptDir = fullfile(projectRoot, 'outputs', 'tapc', ...
        'route_a_backbone_development_v1', 'generation');
    if ~isfolder(receiptDir)
        mkdir(receiptDir);
    end
    diaryPath = fullfile(receiptDir, 'matlab_generation.log');
    diary(diaryPath);
    cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

    addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
        fullfile(generatorRoot, 'generators'));
    cfg = create_closedset_6class_config();
    cfg.seed = 2550000000;
    cfg.output_dir = outputDir;
    cfg.output.save_v73 = true;
    cfg.output.save_legacy = true;
    cfg.experiment = struct( ...
        'name', 'RouteA_BackboneDevelopmentV1', ...
        'evidence_role', 'fresh_train_only_backbone_development', ...
        'generated_test_split', false, ...
        'base_protocol', '0716_closedset_6class');
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_closedset_manifest(cfg, registry, 'train');

    expectedSampleSeeds = [2550000001, 2550025200];
    expectedReceiverSeeds = [2748677743, 2748702942];
    if height(manifest) ~= 25200 || ...
            manifest.sample_seed(1) ~= expectedSampleSeeds(1) || ...
            manifest.sample_seed(end) ~= expectedSampleSeeds(2)
        error('RouteABackbone:ManifestContractFailed', ...
            'The pre-registered sample-seed contract was not reproduced.');
    end

    result = struct( ...
        'status', 'preflight_passed_no_data_generated', ...
        'experiment', 'RouteA_BackboneDevelopmentV1', ...
        'evidence_role', 'fresh_train_only_backbone_development', ...
        'output_dir', outputDir, ...
        'sample_count', height(manifest), ...
        'sample_seed_range', expectedSampleSeeds, ...
        'receiver_noise_seed_range', expectedReceiverSeeds, ...
        'generated_test_split', false, ...
        'locked_test_used', false, ...
        'test_v2_used', false, ...
        'test_v3_opened', false, ...
        'test_v3_used', false);

    if ~executeGeneration
        write_json(fullfile(receiptDir, 'preflight_receipt.json'), result);
        disp(jsonencode(result, PrettyPrint=true));
        return;
    end

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    finalPath = fullfile(outputDir, 'closedset_6class_train.mat');
    partialPath = fullfile(outputDir, 'closedset_6class_train.partial.mat');
    if isfile(partialPath)
        error('RouteABackbone:PartialExists', ...
            ['A partial file already exists. Preserve and inspect it before ' ...
             'any recovery action: %s'], partialPath);
    end
    if isfile(finalPath)
        verification = verify_saved_closedset_split( ...
            finalPath, manifest, cfg, schema);
        result.status = 'completed_existing_verified';
        result.data_path = finalPath;
        result.verification = verification;
        write_json(fullfile(receiptDir, 'generation_receipt.json'), result);
        disp(jsonencode(result, PrettyPrint=true));
        return;
    end

    started = tic;
    bundle = generate_dataset_from_manifest( ...
        cfg, registry, schema, manifest);
    validate_generated_dataset(bundle, cfg, registry, schema, 'balanced');
    statistics = print_dataset_statistics(bundle, registry);
    save_dataset_bundle(partialPath, bundle, cfg, registry, schema);
    clear bundle
    partialVerification = verify_saved_closedset_split( ...
        partialPath, manifest, cfg, schema);
    movefile(partialPath, finalPath);
    finalVerification = verify_saved_closedset_split( ...
        finalPath, manifest, cfg, schema);

    result.status = 'completed_train_only_generation';
    result.data_path = finalPath;
    result.elapsed_seconds = toc(started);
    result.statistics = statistics;
    result.partial_verification = partialVerification;
    result.verification = finalVerification;
    write_json(fullfile(receiptDir, 'generation_receipt.json'), result);
    disp(jsonencode(result, PrettyPrint=true));
end

function write_json(pathValue, payload)
    handle = fopen(pathValue, 'w', 'n', 'UTF-8');
    if handle < 0
        error('RouteABackbone:ReceiptWriteFailed', ...
            'Could not open receipt path: %s', pathValue);
    end
    cleanupHandle = onCleanup(@() fclose(handle)); %#ok<NASGU>
    fprintf(handle, '%s\n', jsonencode(payload, PrettyPrint=true));
end
