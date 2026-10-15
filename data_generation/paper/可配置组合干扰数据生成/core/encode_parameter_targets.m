function [values, mask] = encode_parameter_targets( ...
        schema, registry, activeCodes, componentParams)
%ENCODE_PARAMETER_TARGETS 将活动干扰参数编码到固定长度监督向量。
%   [VALUES, MASK] = ENCODE_PARAMETER_TARGETS(SCHEMA, REGISTRY,
%   ACTIVECODES, COMPONENTPARAMS) 按 SCHEMA 的确定性列顺序返回 1×P
%   double 参数向量 VALUES 和 1×P logical 有效掩码 MASK。ACTIVECODES 是
%   与 COMPONENTPARAMS 一一对应的活动类型代码；每个参数结构体必须包含
%   该类型在 REGISTRY 中登记的全部字段。
%
%   非活动类型、MFT/RangeFakeTarget 未使用的定长槽，以及已经吸收到最终
%   逐目标时延而不可独立监督的 arrival_delay_s 均保持 NaN/false。只有
%   有限实数标量才写入 VALUES 并置 MASK=true；NaN 表示有意关闭监督，
%   Inf、复数、向量或字段不一致均作为编码错误拒绝。

    validate_inputs(schema, registry, activeCodes, componentParams);
    activeCodes = normalize_codes(activeCodes);
    componentParams = reshape(componentParams, 1, []);
    values = nan(1, height(schema));
    mask = false(1, height(schema));
    registryCodes = {registry.code};

    for componentIndex = 1:numel(activeCodes)
        code = activeCodes{componentIndex};
        registryIndex = find(strcmp(code, registryCodes), 1, 'first');
        entry = registry(registryIndex);
        params = componentParams{componentIndex};
        expectedNames = reshape(entry.parameter_names, 1, []);
        actualNames = reshape(fieldnames(params), 1, []);
        if ~isequal(sort(actualNames), sort(expectedNames))
            throw_invalid_parameter_encoding( ...
                sprintf('类型 %s 的参数字段集合与注册表不一致。', code));
        end

        for parameterIndex = 1:numel(expectedNames)
            parameterName = expectedNames{parameterIndex};
            fullName = sprintf('%s.%s', code, parameterName);
            schemaRow = find(strcmp(schema.field_name, fullName));
            if numel(schemaRow) ~= 1 || ...
                    ~strcmp(schema.owner_code{schemaRow}, code)
                throw_invalid_parameter_encoding( ...
                    sprintf('参数模式缺少唯一字段 %s。', fullName));
            end
            value = params.(parameterName);
            if ~isnumeric(value) || ~isreal(value) || ~isscalar(value)
                throw_invalid_parameter_encoding( ...
                    sprintf('%s 必须是实数数值标量或 NaN。', fullName));
            end
            value = double(value);
            if isfinite(value)
                values(schemaRow) = value;
                mask(schemaRow) = true;
            elseif ~isnan(value)
                throw_invalid_parameter_encoding( ...
                    sprintf('%s 不能是 Inf 或 -Inf。', fullName));
            end
        end
    end
end

function validate_inputs(schema, registry, activeCodes, componentParams)
%VALIDATE_INPUTS 校验参数模式、注册表和活动分量容器的基本结构。

    requiredColumns = {'column_index', 'field_name', 'owner_code', 'unit'};
    if ~istable(schema) || ~all(ismember(requiredColumns, ...
            schema.Properties.VariableNames)) || height(schema) < 1
        throw_invalid_parameter_encoding('schema 不是合法参数模式表。');
    end
    if ~isstruct(registry) || isempty(registry) || ...
            ~all(isfield(registry, {'code', 'parameter_names'}))
        throw_invalid_parameter_encoding('registry 不是合法注册表。');
    end
    normalizedCodes = normalize_codes(activeCodes);
    if ~iscell(componentParams) || ...
            numel(componentParams) ~= numel(normalizedCodes) || ...
            ~all(cellfun(@(value) isstruct(value) && isscalar(value), ...
            componentParams))
        throw_invalid_parameter_encoding( ...
            'componentParams 必须是与 activeCodes 等长的标量结构体单元数组。');
    end
    if numel(unique(normalizedCodes, 'stable')) ~= numel(normalizedCodes)
        throw_invalid_parameter_encoding('activeCodes 不能包含重复类型。');
    end
    registryCodes = {registry.code};
    if ~all(ismember(normalizedCodes, registryCodes))
        throw_invalid_parameter_encoding('activeCodes 含注册表未定义类型。');
    end
end

function codes = normalize_codes(value)
%NORMALIZE_CODES 将 cellstr 或 string 数组规范为 1×K cellstr。

    if isstring(value) && ~any(ismissing(value), 'all')
        codes = cellstr(value(:).');
    elseif iscellstr(value)
        codes = reshape(value, 1, []);
    else
        throw_invalid_parameter_encoding( ...
            'activeCodes 必须是非缺失 string 数组或 cellstr。');
    end
    codes = cellfun(@strtrim, codes, 'UniformOutput', false);
    if isempty(codes) || any(cellfun(@isempty, codes))
        throw_invalid_parameter_encoding('activeCodes 不能为空。');
    end
end

function throw_invalid_parameter_encoding(message)
%THROW_INVALID_PARAMETER_ENCODING 抛出统一参数编码错误标识。

    error('JammingComposition:InvalidParameterEncoding', '%s', message);
end
