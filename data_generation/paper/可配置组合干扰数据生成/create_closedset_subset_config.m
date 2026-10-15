function cfg = create_closedset_subset_config( ...
        selectedTypes, seed, outputDir)
%CREATE_CLOSEDSET_SUBSET_CONFIG 创建全组合闭集类型子集配置。
%   CFG = CREATE_CLOSEDSET_SUBSET_CONFIG(SELECTEDTYPES, SEED, OUTPUTDIR)
%   复用已经验收的六类雷达体制和各单类物理参数，只替换实际参与组合的
%   已知类型、独立实验种子、输出目录和类别相关规模。SELECTEDTYPES 中的
%   全部类型均为已知类，未知类为空；组合模式固定为all，每个非空组合
%   生成400条，因此单个split样本数为(2^类别数-1)*400。
%
%   本配置用于10--30 dB全功率覆盖场景。每个活动分量从公共JNR范围独立
%   抽样，不声明与九挡数据配对。experiment.full_power随配置写入MAT，
%   供增强回载核对JNR范围、目标功率、实测功率及参数监督一致性。

    selectedTypes = validate_selected_types(selectedTypes);
    seed = validate_seed(seed);
    outputDir = validate_output_dir(outputDir);

    % 直接继承六类正式配置，避免雷达参数和类型参数出现第二套定义。
    cfg = create_closedset_6class_config();
    registry = build_jamming_registry(cfg);
    validCodes = {registry.code};
    invalidCodes = selectedTypes(~ismember(selectedTypes, validCodes));
    if ~isempty(invalidCodes)
        error('JammingSubset:InvalidTypes', ...
            'selectedTypes包含未注册类型：%s。', ...
            strjoin(invalidCodes, '、'));
    end

    classCount = numel(selectedTypes);
    patternCount = 2 ^ classCount - 1;
    cfg.seed = seed;
    cfg.output_dir = outputDir;
    cfg.output.dataset_stem = sprintf('closedset_%dclass', classCount);
    cfg.types.enabled = selectedTypes;
    cfg.types.known = selectedTypes;
    cfg.types.unknown = {};
    cfg.dataset.num_samples = patternCount * 400;
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = classCount;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 400;
    cfg.params.common.JNR_dB = [10, 30];

    % 保存实验语义快照，验证器据此区分连续全功率和确定性九挡功率。
    cfg.experiment = struct();
    cfg.experiment.name = sprintf( ...
        'closedset_%dclass_full_power_10_30dB_v1', classCount);
    cfg.experiment.full_power = struct( ...
        'JNR_range_dB', [10, 30], ...
        'independent_components', true);
end

function selectedTypes = validate_selected_types(value)
%VALIDATE_SELECTED_TYPES 规范非空、无重复的字符行单元。

    if isstring(value) && isrow(value) && all(~ismissing(value))
        value = cellstr(value);
    end
    valid = iscell(value) && isrow(value) && ~isempty(value) && ...
        all(cellfun(@(item) ischar(item) && isrow(item) && ...
        ~isempty(strtrim(item)), value));
    if ~valid
        error('JammingSubset:InvalidTypes', ...
            'selectedTypes必须是非空字符行单元。');
    end
    selectedTypes = cellfun(@strtrim, value, 'UniformOutput', false);
    if numel(unique(selectedTypes, 'stable')) ~= numel(selectedTypes)
        error('JammingSubset:InvalidTypes', ...
            'selectedTypes不允许包含重复类型。');
    end
end

function seed = validate_seed(value)
%VALIDATE_SEED 规范MATLAB随机流可接受的非负整数种子。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value >= 0 && value <= 2 ^ 32 - 1 && ...
        value == floor(value);
    if ~valid
        error('JammingSubset:InvalidSeed', ...
            'seed必须是0到2^32-1之间的整数标量。');
    end
    seed = double(value);
end

function outputDir = validate_output_dir(value)
%VALIDATE_OUTPUT_DIR 规范非空输出目录文本但不创建目录。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        error('JammingSubset:InvalidOutputDir', ...
            'outputDir必须是非空文本标量。');
    end
    outputDir = strtrim(value);
    if isempty(outputDir)
        error('JammingSubset:InvalidOutputDir', ...
            'outputDir不能为空。');
    end
end
