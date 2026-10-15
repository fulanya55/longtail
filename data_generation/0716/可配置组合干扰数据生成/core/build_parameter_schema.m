function schema = build_parameter_schema(registry)
%BUILD_PARAMETER_SCHEMA 构建参数掩码使用的确定性列模式。
%   SCHEMA = BUILD_PARAMETER_SCHEMA(REGISTRY) 按 REGISTRY 的条目顺序及每个
%   条目 parameter_names 的声明顺序，返回列严格为 column_index、
%   field_name、owner_code、unit 的 table。field_name 使用
%   '<TYPE>.<parameter_name>'，便于参数向量、parameter mask 和有效列索引
%   在训练、保存与读取阶段保持一一对齐。
%
%   REGISTRY 中 parameter_units 的单位原样写入 unit，例如 dB、Hz、Hz/s、
%   s、m、m/s；单位 1 表示计数、模式代码或无量纲量。函数不写文件、不
%   修改注册表，也不改变随机数状态。
%
%   示例：
%       registry = build_jamming_registry();
%       schema = build_parameter_schema(registry);
%       firstMaskColumn = schema.field_name{1};

    if ~isstruct(registry) || isempty(registry)
        error('JammingSchema:InvalidRegistryEntry', ...
            'registry 必须是非空结构体数组。');
    end
    requiredFields = {'code', 'parameter_names', 'parameter_units'};
    if ~all(isfield(registry, requiredFields))
        error('JammingSchema:InvalidRegistryEntry', ...
            '每个 registry 条目必须包含 code、parameter_names 和 parameter_units。');
    end

    entryCount = numel(registry);
    normalizedCodes = cell(1, entryCount);
    normalizedNames = cell(1, entryCount);
    normalizedUnits = cell(1, entryCount);
    parameterCounts = zeros(1, entryCount);
    for registryIndex = 1:entryCount
        entry = registry(registryIndex);
        normalizedCodes{registryIndex} = normalize_text_scalar( ...
            entry.code, sprintf('registry(%d).code', registryIndex));
        normalizedNames{registryIndex} = normalize_text_collection( ...
            entry.parameter_names, ...
            sprintf('registry(%d).parameter_names', registryIndex));
        normalizedUnits{registryIndex} = normalize_text_collection( ...
            entry.parameter_units, ...
            sprintf('registry(%d).parameter_units', registryIndex));
        if numel(normalizedNames{registryIndex}) ~= ...
                numel(normalizedUnits{registryIndex})
            error('JammingSchema:InvalidRegistryEntry', ...
                'registry(%d) 的参数名和单位数量必须相等。', registryIndex);
        end
        parameterCounts(registryIndex) = ...
            numel(normalizedNames{registryIndex});
    end

    rowCount = sum(parameterCounts);
    fieldName = cell(rowCount, 1);
    ownerCode = cell(rowCount, 1);
    unit = cell(rowCount, 1);

    rowIndex = 0;
    for registryIndex = 1:entryCount
        parameterNames = normalizedNames{registryIndex};
        parameterUnits = normalizedUnits{registryIndex};
        for parameterIndex = 1:numel(parameterNames)
            rowIndex = rowIndex + 1;
            ownerCode{rowIndex} = normalizedCodes{registryIndex};
            fieldName{rowIndex} = sprintf('%s.%s', ownerCode{rowIndex}, ...
                parameterNames{parameterIndex});
            unit{rowIndex} = parameterUnits{parameterIndex};
        end
    end

    [~, firstOccurrence] = unique(fieldName, 'stable');
    if numel(firstOccurrence) ~= rowCount
        duplicateRows = setdiff((1:rowCount)', firstOccurrence, 'stable');
        error('JammingSchema:DuplicateField', ...
            '参数模式含重复完整字段名：%s。', fieldName{duplicateRows(1)});
    end

    columnIndex = (1:rowCount)';
    schema = table(columnIndex, fieldName, ownerCode, unit, ...
        'VariableNames', {'column_index', 'field_name', ...
        'owner_code', 'unit'});
end

function textValue = normalize_text_scalar(value, path)
%NORMALIZE_TEXT_SCALAR 校验并规范一个非空文本标量。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        textValue = char(value);
    elseif ischar(value) && isrow(value)
        textValue = value;
    else
        error('JammingSchema:InvalidRegistryEntry', ...
            '%s 必须是非空字符行向量或 string 标量。', path);
    end
    textValue = strtrim(textValue);
    if isempty(textValue)
        error('JammingSchema:InvalidRegistryEntry', ...
            '%s 不能是空文本。', path);
    end
end

function textValues = normalize_text_collection(value, path)
%NORMALIZE_TEXT_COLLECTION 校验 cellstr 或 string 文本集合。

    if isstring(value)
        if any(ismissing(value), 'all')
            error('JammingSchema:InvalidRegistryEntry', ...
                '%s 不能含缺失 string。', path);
        end
        textValues = cellstr(value(:).');
    elseif iscellstr(value)
        textValues = reshape(value, 1, []);
    else
        error('JammingSchema:InvalidRegistryEntry', ...
            '%s 必须是 cellstr 或 string 文本集合。', path);
    end
    textValues = cellfun(@strtrim, textValues, 'UniformOutput', false);
    if any(cellfun(@isempty, textValues))
        error('JammingSchema:InvalidRegistryEntry', ...
            '%s 的每个文本元素都必须非空。', path);
    end
end
