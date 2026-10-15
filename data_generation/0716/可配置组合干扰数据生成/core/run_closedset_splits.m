function result = run_closedset_splits(cfg, registry, schema, ...
        executeGeneration, splitSelection, existingPolicy)
%RUN_CLOSEDSET_SPLITS 按选择和已有文件策略原子生成闭集训练/测试划分。
%   RESULT = RUN_CLOSEDSET_SPLITS(CFG, REGISTRY, SCHEMA, EXECUTEGENERATION,
%   SPLITSELECTION, EXISTINGPOLICY) 是六类公共入口使用的可测试执行内核。
%   SPLITSELECTION 允许 both、train、test；EXISTINGPOLICY 允许：
%     resume    最终文件存在时先做增强回载，通过则复用，否则报错；
%     error     最终文件存在时立即报 JammingDataset:OutputAlreadyExists；
%     overwrite 只有该策略允许以新文件替换已有最终文件。
%
%   正式生成先写同目录 *.partial.mat，释放大型 bundle 后校验 partial，
%   再用 MOVEFILE 提交到最终路径并二次校验。生成中断最多留下 partial，
%   不会把未完成文件伪装成最终文件。启动新生成前仅删除当前划分精确同名
%   stale partial，不使用通配符，也不触碰其他文件。

    executeGeneration = normalize_execute_flag(executeGeneration);
    splitSelection = normalize_choice(splitSelection, ...
        {'both', 'train', 'test'}, ...
        'JammingDataset:InvalidSplitSelection', 'splitSelection');
    existingPolicy = normalize_choice(existingPolicy, ...
        {'resume', 'error', 'overwrite'}, ...
        'JammingDataset:InvalidExistingPolicy', 'existingPolicy');
    validate_runner_inputs(cfg, registry, schema);

    trainPath = fullfile(cfg.output_dir, 'closedset_6class_train.mat');
    testPath = fullfile(cfg.output_dir, 'closedset_6class_test.mat');
    trainPartialPath = fullfile( ...
        cfg.output_dir, 'closedset_6class_train.partial.mat');
    testPartialPath = fullfile( ...
        cfg.output_dir, 'closedset_6class_test.partial.mat');

    result = struct();
    result.executed = executeGeneration;
    result.split_selection = splitSelection;
    result.existing_policy = existingPolicy;
    result.config = cfg;
    result.registry = registry;
    result.parameter_schema = schema;
    result.train_path = trainPath;
    result.test_path = testPath;
    result.splits = struct( ...
        'train', initialize_split_result(trainPath, trainPartialPath), ...
        'test', initialize_split_result(testPath, testPartialPath));

    selectedSplits = resolve_selected_splits(splitSelection);
    if executeGeneration && ~isfolder(cfg.output_dir)
        mkdir(cfg.output_dir);
    end

    % 始终按 train、test 顺序循环；both 时保证训练划分完全提交并释放内存
    % 后才开始测试划分。test 选择会直接跳过 train。
    orderedSplits = {'train', 'test'};
    for splitIndex = 1:numel(orderedSplits)
        splitName = orderedSplits{splitIndex};
        if ~ismember(splitName, selectedSplits)
            result.splits.(splitName).status = 'not_selected';
            result.splits.(splitName).skipped = true;
            result.splits.(splitName).verification = struct( ...
                'performed', false, 'reason', 'not_selected');
            continue;
        end

        result.splits.(splitName).requested = true;
        fprintf('[闭集数据] 构建 %s manifest...\n', splitName);
        manifest = build_closedset_manifest(cfg, registry, splitName);
        summary = summarize_manifest(manifest, ...
            cfg.dataset.samples_per_pattern);
        result.splits.(splitName).summary = summary;
        result.([splitName '_summary']) = summary;

        if ~executeGeneration
            result.splits.(splitName).status = 'dry_run';
            result.splits.(splitName).skipped = true;
            result.splits.(splitName).verification = struct( ...
                'performed', false, 'reason', 'dry_run');
            result.([splitName '_manifest']) = manifest;
            fprintf('[闭集数据] %s 干运行完成：%d 条，%d 个模式。\n', ...
                splitName, summary.sample_count, summary.pattern_count);
            continue;
        end

        finalPath = result.splits.(splitName).path;
        partialPath = result.splits.(splitName).partial_path;
        splitResult = execute_one_split(cfg, registry, schema, manifest, ...
            splitName, finalPath, partialPath, existingPolicy, summary);
        result.splits.(splitName) = splitResult;
        result.([splitName '_summary']) = splitResult.summary;
    end
end

function splitResult = execute_one_split(cfg, registry, schema, manifest, ...
        splitName, finalPath, partialPath, existingPolicy, summary)
%EXECUTE_ONE_SPLIT 处理单个划分的复用、原子保存和双重回载。

    splitResult = initialize_split_result(finalPath, partialPath);
    splitResult.requested = true;
    splitResult.summary = summary;

    if isfile(finalPath)
        switch existingPolicy
            case 'error'
                error('JammingDataset:OutputAlreadyExists', ...
                    '最终文件已存在，error 策略拒绝继续：%s。', finalPath);
            case 'resume'
                fprintf('[闭集数据] %s 最终文件存在，正在增强回载验证...\n', ...
                    splitName);
                if isfile(partialPath)
                    fprintf(['[闭集数据] 检测到同名旧 partial；先保留现场，' ...
                        '最终文件验证通过后再清理。\n']);
                end
                verification = verify_saved_closedset_split( ...
                    finalPath, manifest, cfg, schema);
                cleanup_stale_partial(partialPath);
                splitResult.status = 'reused';
                splitResult.reused = true;
                splitResult.skipped = true;
                splitResult.verification = verification;
                fprintf('[闭集数据] %s 已验证并复用，跳过生成。\n', splitName);
                return;
            case 'overwrite'
                % 只有 overwrite 会在原子提交阶段替换已有最终文件。
        end
    end

    cleanup_stale_partial(partialPath);
    fprintf('[闭集数据] %s 开始生成 %d 条样本...\n', ...
        splitName, summary.sample_count);
    bundle = generate_dataset_from_manifest(cfg, registry, schema, manifest);
    validate_generated_dataset(bundle, cfg, registry, schema, 'balanced');
    statistics = print_dataset_statistics(bundle, registry);
    fprintf('[闭集数据] %s 保存到临时文件：%s\n', splitName, partialPath);
    save_dataset_bundle(partialPath, bundle, cfg, registry, schema);

    % 必须先释放数 GB 的内存 bundle，再回载 partial；函数异常退出时局部
    % 变量也会释放，而已写出的 partial 会保留供诊断，不会冒充 final。
    clear bundle
    partialVerification = verify_saved_closedset_split( ...
        partialPath, manifest, cfg, schema);
    commit_partial_file(partialPath, finalPath, existingPolicy);
    finalVerification = verify_saved_closedset_split( ...
        finalPath, manifest, cfg, schema);

    summary.statistics = statistics;
    splitResult.status = 'generated';
    splitResult.generated = true;
    splitResult.summary = summary;
    splitResult.partial_verification = partialVerification;
    splitResult.verification = finalVerification;
    fprintf('[闭集数据] %s 已原子提交并通过最终回载。\n', splitName);
end

function commit_partial_file(partialPath, finalPath, existingPolicy)
%COMMIT_PARTIAL_FILE 将已验证 partial 原子式移动为最终文件。

    if ~isfile(partialPath)
        error('JammingDataset:AtomicCommitFailed', ...
            '待提交的 partial 文件不存在：%s。', partialPath);
    end
    if strcmp(existingPolicy, 'overwrite')
        [success, message] = movefile(partialPath, finalPath, 'f');
    else
        if isfile(finalPath)
            error('JammingDataset:OutputAlreadyExists', ...
                '提交前最终文件已出现，拒绝覆盖：%s。', finalPath);
        end
        [success, message] = movefile(partialPath, finalPath);
    end
    if ~success
        error('JammingDataset:AtomicCommitFailed', ...
            '无法把 partial 提交为最终文件：%s', message);
    end
end

function cleanup_stale_partial(partialPath)
%CLEANUP_STALE_PARTIAL 只删除当前划分精确同名的旧 partial 文件。

    if ~isfile(partialPath)
        return;
    end
    fprintf('[闭集数据] 删除旧 partial：%s\n', partialPath);
    try
        delete(partialPath);
    catch exception
        error('JammingDataset:PartialCleanupFailed', ...
            '无法删除旧 partial 文件 %s：%s', partialPath, exception.message);
    end
    if isfile(partialPath)
        error('JammingDataset:PartialCleanupFailed', ...
            '旧 partial 文件删除后仍然存在：%s。', partialPath);
    end
end

function splitResult = initialize_split_result(finalPath, partialPath)
%INITIALIZE_SPLIT_RESULT 创建固定字段的轻量划分状态。

    splitResult = struct( ...
        'requested', false, ...
        'status', 'pending', ...
        'generated', false, ...
        'reused', false, ...
        'skipped', false, ...
        'path', finalPath, ...
        'partial_path', partialPath, ...
        'summary', struct(), ...
        'verification', struct('performed', false, 'reason', 'pending'));
end

function selectedSplits = resolve_selected_splits(splitSelection)
%RESOLVE_SELECTED_SPLITS 把选择代码展开为稳定顺序的划分列表。

    switch splitSelection
        case 'both'
            selectedSplits = {'train', 'test'};
        case 'train'
            selectedSplits = {'train'};
        case 'test'
            selectedSplits = {'test'};
    end
end

function value = normalize_execute_flag(value)
%NORMALIZE_EXECUTE_FLAG 把逻辑标量或数值 0/1 规范为逻辑标量。

    validLogical = islogical(value) && isscalar(value);
    validNumeric = isnumeric(value) && isscalar(value) && isreal(value) && ...
        isfinite(value) && any(value == [0, 1]);
    if ~(validLogical || validNumeric)
        error('JammingDataset:InvalidExecutionFlag', ...
            'executeGeneration 必须是逻辑标量或数值 0/1。');
    end
    value = logical(value);
end

function value = normalize_choice(value, allowed, errorId, path)
%NORMALIZE_CHOICE 规范选择文本并验证其属于允许集合。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        error(errorId, '%s 必须是文本标量。', path);
    end
    value = lower(strtrim(value));
    if ~ismember(value, allowed)
        error(errorId, '%s 仅允许：%s。', path, strjoin(allowed, ', '));
    end
end

function validate_runner_inputs(cfg, registry, schema)
%VALIDATE_RUNNER_INPUTS 防御性校验执行内核输入。

    validCfg = isstruct(cfg) && isscalar(cfg) && ...
        isfield(cfg, 'output_dir') && isfield(cfg, 'derived') && ...
        isfield(cfg, 'dataset');
    if ~validCfg || ~(ischar(cfg.output_dir) || ...
            (isstring(cfg.output_dir) && isscalar(cfg.output_dir)))
        error('JammingDataset:InvalidRunnerInput', ...
            'cfg 必须是含输出目录、派生字段和数据集字段的标准化配置。');
    end
    if ~isstruct(registry) || isempty(registry) || ...
            ~all(isfield(registry, {'code', 'generator'}))
        error('JammingDataset:InvalidRunnerInput', 'registry 不是合法注册表。');
    end
    if ~istable(schema) || height(schema) < 1
        error('JammingDataset:InvalidRunnerInput', 'schema 不是合法参数模式表。');
    end
end

function summary = summarize_manifest(manifest, samplesPerPattern)
%SUMMARIZE_MANIFEST 返回不依赖波形内容的轻量清单摘要。

    summary = struct( ...
        'split', manifest.split{1}, ...
        'sample_count', height(manifest), ...
        'pattern_count', numel(unique(manifest.combination_id)), ...
        'samples_per_pattern', samplesPerPattern);
end
