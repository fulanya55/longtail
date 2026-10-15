function result = run_subset_dataset_family( ...
        fullCfg, sweepConfigs, executeGeneration, datasetSelection, ...
        splitSelection, existingPolicy)
%RUN_SUBSET_DATASET_FAMILY 串行执行一个类别数的两种功率数据。
%   RESULT = RUN_SUBSET_DATASET_FAMILY(FULLCFG, SWEEPCONFIGS,
%   EXECUTEGENERATION, DATASETSELECTION, SPLITSELECTION, EXISTINGPOLICY)
%   按“全功率后九挡”的稳定顺序复用已有闭集runner。DATASETSELECTION
%   允许all、full_power、power_sweep；未选择的部分返回固定轻量状态。
%   dry-run不创建目录、CSV或MAT，正式运行仍由底层runner执行partial、
%   双重回载和resume语义。

    executeGeneration = normalize_execute_flag(executeGeneration);
    datasetSelection = normalize_dataset_selection(datasetSelection);
    result = struct();
    result.executed = executeGeneration;
    result.dataset_selection = datasetSelection;
    result.full_power = unselected_result();
    result.power_sweep = unselected_result();

    if ismember(datasetSelection, {'all', 'full_power'})
        cfg = normalize_and_validate_config(fullCfg, 'balanced');
        registry = build_jamming_registry(cfg);
        schema = build_parameter_schema(registry);
        result.full_power = run_closedset_splits( ...
            cfg, registry, schema, executeGeneration, ...
            splitSelection, existingPolicy);
    end

    if ismember(datasetSelection, {'all', 'power_sweep'})
        result.power_sweep = run_power_sweep_levels( ...
            sweepConfigs, executeGeneration, splitSelection, ...
            existingPolicy, executeGeneration);
    end
end

function value = normalize_execute_flag(value)
%NORMALIZE_EXECUTE_FLAG 规范逻辑或数值0/1执行开关。

    valid = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
        isreal(value) && isfinite(double(value)) && ...
        ismember(double(value), [0, 1]);
    if ~valid
        error('JammingDataset:InvalidExecutionFlag', ...
            'executeGeneration必须是逻辑标量或数值0/1。');
    end
    value = logical(value);
end

function value = normalize_dataset_selection(value)
%NORMALIZE_DATASET_SELECTION 规范数据种类选择文本。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        error('JammingSubset:InvalidDatasetSelection', ...
            'datasetSelection必须是文本标量。');
    end
    value = lower(strtrim(value));
    if ~ismember(value, {'all', 'full_power', 'power_sweep'})
        error('JammingSubset:InvalidDatasetSelection', ...
            'datasetSelection仅允许all、full_power或power_sweep。');
    end
end

function value = unselected_result()
%UNSELECTED_RESULT 创建未选择数据种类的稳定轻量结果。

    value = struct('executed', false, 'status', 'not_selected');
end
