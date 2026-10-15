function truthPath = save_legacy_truth_sidecar( ...
        mainFilePath, bundle, saveV73)
%SAVE_LEGACY_TRUTH_SIDECAR 保存与历史主文件同 stem 的真值旁车。
%   TRUTHPATH = SAVE_LEGACY_TRUTH_SIDECAR(MAINFILEPATH, BUNDLE, SAVEV73)
%   将 BUNDLE 字段保存为 MAT 顶层变量。输出路径为
%   <main-stem>_truth.mat；历史主文件本身不会被修改。
%
%   SAVEV73 省略时默认为 true。写文件前会核对必需字段、行数、NaN/mask
%   形状和固定 79 维宽度。输入错误抛出 LegacyJamming:InvalidTruthShape，
%   实际保存失败抛出 LegacyJamming:SidecarSaveFailed。

    if nargin < 3
        saveV73 = true;
    end
    if isstring(mainFilePath) && isscalar(mainFilePath) && ...
            ~ismissing(mainFilePath)
        mainFilePath = char(mainFilePath);
    elseif ~(ischar(mainFilePath) && isrow(mainFilePath))
        error('LegacyJamming:InvalidTruthShape', ...
            'mainFilePath 必须是文本标量。');
    end
    if ~(islogical(saveV73) && isscalar(saveV73))
        error('LegacyJamming:InvalidTruthShape', ...
            'saveV73 必须是 logical 标量。');
    end
    requiredFields = {'parameter_values', 'parameter_mask', ...
        'parameter_schema', 'label', 'legacy_class_codes', ...
        'truth_coverage', 'atomic_truth', 'source_sample_index', ...
        'metadata'};
    validBundle = isstruct(bundle) && isscalar(bundle) && ...
        all(isfield(bundle, requiredFields)) && ...
        isnumeric(bundle.parameter_values) && ...
        size(bundle.parameter_values, 2) == 79 && ...
        isequal(size(bundle.parameter_values), size(bundle.parameter_mask)) && ...
        size(bundle.parameter_values, 1) == size(bundle.label, 1) && ...
        numel(bundle.source_sample_index) == size(bundle.label, 1) && ...
        istable(bundle.parameter_schema) && ...
        height(bundle.parameter_schema) == 79;
    if ~validBundle
        error('LegacyJamming:InvalidTruthShape', ...
            '旁车 bundle 字段、行数或 79 维宽度不合法。');
    end

    [parent, stem, extension] = fileparts(strtrim(mainFilePath));
    if isempty(stem) || ~strcmpi(extension, '.mat')
        error('LegacyJamming:InvalidTruthShape', ...
            '历史主文件路径必须以 .mat 结尾。');
    end
    truthPath = fullfile(parent, [stem '_truth.mat']);
    saveData = bundle;
    try
        if saveV73
            save(truthPath, '-struct', 'saveData', '-v7.3');
        else
            save(truthPath, '-struct', 'saveData');
        end
    catch saveError
        error('LegacyJamming:SidecarSaveFailed', ...
            '真值旁车保存失败：%s', saveError.message);
    end
    if ~isfile(truthPath)
        error('LegacyJamming:SidecarSaveFailed', ...
            '真值旁车文件未成功创建。');
    end
end
