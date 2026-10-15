function save_dataset_bundle(outputPath, bundle, cfg, registry, schema)
%SAVE_DATASET_BUNDLE 将数据包字段保存为 MAT 顶层变量。
%   SAVE_DATASET_BUNDLE(OUTPUTPATH, BUNDLE, CFG, REGISTRY, SCHEMA) 不把
%   全部数据隐藏在单一嵌套结构中，而是使用 save -struct 写出
%   iq_complex、标签、参数、manifest 和 metadata 等顶层变量，并额外
%   写出 config、registry_snapshot、parameter_schema 方便独立读取。
%   cfg.output.save_v73=true 时使用 -v7.3，测试和小文件可关闭该选项。

    outputPath = normalize_output_path(outputPath);
    if ~isstruct(bundle) || ~isscalar(bundle)
        error('JammingDataset:InvalidSaveInput', 'bundle 必须是标量结构体。');
    end
    if ~isfield(bundle, 'metadata') || ...
            ~isstruct(bundle.metadata) || ~isscalar(bundle.metadata) || ...
            ~isfield(bundle.metadata, 'scientific_semantics_version') || ...
            ~strcmp(bundle.metadata.scientific_semantics_version, ...
            current_scientific_semantics_version())
        error('JammingDataset:ScientificSemanticsMismatch', ...
            '只允许保存带当前科学语义版本的数据包。');
    end
    parent = fileparts(outputPath);
    if ~isempty(parent) && ~isfolder(parent)
        mkdir(parent);
    end

    saveData = bundle;
    saveData.config = cfg;
    saveData.registry_snapshot = make_registry_snapshot(registry);
    saveData.parameter_schema = schema;
    if cfg.output.save_v73
        save(outputPath, '-struct', 'saveData', '-v7.3');
    else
        save(outputPath, '-struct', 'saveData');
    end
    if ~isfile(outputPath)
        error('JammingDataset:SaveFailed', 'MAT 文件未成功创建。');
    end
end

function outputPath = normalize_output_path(value)
%NORMALIZE_OUTPUT_PATH 校验非空 MAT 输出路径文本。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        outputPath = char(value);
    elseif ischar(value) && isrow(value)
        outputPath = value;
    else
        error('JammingDataset:InvalidSaveInput', ...
            'outputPath 必须是非空文本标量。');
    end
    outputPath = strtrim(outputPath);
    if isempty(outputPath)
        error('JammingDataset:InvalidSaveInput', 'outputPath 不能为空。');
    end
    [~, ~, extension] = fileparts(outputPath);
    if isempty(extension)
        outputPath = [outputPath '.mat'];
    elseif ~strcmpi(extension, '.mat')
        error('JammingDataset:InvalidSaveInput', ...
            'outputPath 扩展名必须是 .mat。');
    end
end

function snapshot = make_registry_snapshot(registry)
%MAKE_REGISTRY_SNAPSHOT 移除函数句柄并保留生成器函数名称。

    if ~isstruct(registry) || isempty(registry) || ...
            ~all(isfield(registry, {'generator', 'code'}))
        error('JammingDataset:InvalidSaveInput', 'registry 不是合法注册表。');
    end
    names = arrayfun(@(entry) func2str(entry.generator), ...
        registry, 'UniformOutput', false);
    snapshot = rmfield(registry, 'generator');
    for index = 1:numel(snapshot)
        snapshot(index).generator_name = names{index};
    end
end
