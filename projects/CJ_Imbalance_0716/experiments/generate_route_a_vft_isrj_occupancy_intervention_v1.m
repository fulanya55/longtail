function receipt = generate_route_a_vft_isrj_occupancy_intervention_v1(executeGeneration)
%GENERATE_ROUTE_A_VFT_ISRJ_OCCUPANCY_INTERVENTION_V1 Fresh paired regimes.
    if nargin < 1
        executeGeneration = false;
    end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0801_vft_isrj_occupancy_v1';
    auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_vft_isrj_occupancy_intervention_v1_generation');
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));

    names = {'W10_15_R3_4', 'W22_30_R3_4', 'W10_15_R6_8', 'W22_30_R6_8'};
    widths = {[10e-6,15e-6], [22e-6,30e-6], [10e-6,15e-6], [22e-6,30e-6]};
    replicas = {[3,4], [3,4], [6,8], [6,8]};
    baseSeed = 178000000;
    results = cell(numel(names), 1);
    for index = 1:numel(names)
        cfg = create_closedset_6class_config();
        cfg.seed = baseSeed;
        cfg.output_dir = fullfile(outputRoot, names{index});
        cfg.output.save_legacy = false;
        cfg.dataset.samples_per_pattern = 20;
        cfg.dataset.num_samples = 1260;
        cfg.params.ISRJ.sample_width_s = widths{index};
        cfg.params.ISRJ.num_replicas = replicas{index};
        cfg = normalize_and_validate_config(cfg, 'balanced');
        registry = build_jamming_registry(cfg);
        schema = build_parameter_schema(registry);
        results{index} = run_closedset_splits(cfg, registry, schema, logical(executeGeneration), 'train', 'error');
    end
    receipt = struct('status', 'completed', 'executed', logical(executeGeneration), ...
        'experiment', 'RouteA_VFTISRJ_OccupancyIntervention_v1', ...
        'evidence_role', 'fresh_paired_external_physics_diagnostic', ...
        'base_seed', baseSeed, 'regime_names', {names}, ...
        'sample_width_s', {widths}, 'num_replicas', {replicas}, ...
        'samples_per_pattern', 20, 'sample_count_per_regime', 1260, ...
        'split_selection', 'train_only', 'test_split_generated', false, ...
        'locked_test_used', false, 'test_v2_used', false, 'test_v3_used', false);
    if executeGeneration
        receipt.regime_status = cellfun(@(r) r.splits.train.status, results, 'UniformOutput', false);
        receipt.train_paths = cellfun(@(r) r.train_path, results, 'UniformOutput', false);
    end
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix = 'dry_run'; if executeGeneration; prefix = 'formal'; end
    receiptPath = fullfile(auditRoot, sprintf('%s_generation_receipt.json', prefix));
    fid = fopen(receiptPath, 'w', 'n', 'UTF-8');
    if fid < 0; error('RouteA:ReceiptWriteFailed', 'Cannot write receipt.'); end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char');
    fprintf('ROUTE_A_VFT_ISRJ_OCCUPANCY_RECEIPT=%s\n', receiptPath);
end
