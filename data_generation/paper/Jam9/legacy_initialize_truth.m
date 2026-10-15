function truth = legacy_initialize_truth(sampleCount, fieldNames)
%LEGACY_INITIALIZE_TRUTH 创建逐样本 NaN 真值结构。
%   TRUTH = LEGACY_INITIALIZE_TRUTH(SAMPLECOUNT, FIELDNAMES) 返回标量
%   结构体。FIELDNAMES 中的每个字段均为 SAMPLECOUNT×1 double 列向量，
%   初值全部为 NaN。未被最终波形唯一确定的量应继续保持 NaN，后续参数
%   编码器据此保持对应 parameter_mask=false。
%
%   SAMPLECOUNT 必须为有限正整数。FIELDNAMES 必须是非空 cellstr，字段名
%   必须合法且互不重复。输入不合法时抛出稳定错误
%   LegacyJamming:InvalidTruthShape。

    validCount = isnumeric(sampleCount) && isreal(sampleCount) && ...
        isscalar(sampleCount) && isfinite(sampleCount) && ...
        sampleCount >= 1 && sampleCount == fix(sampleCount);
    validNames = iscellstr(fieldNames) && ~isempty(fieldNames) && ...
        numel(unique(fieldNames)) == numel(fieldNames) && ...
        all(cellfun(@isvarname, fieldNames));
    if ~(validCount && validNames)
        error('LegacyJamming:InvalidTruthShape', ...
            ['sampleCount 必须是有限正整数，fieldNames 必须是非空、' ...
            '唯一且合法的字段名单。']);
    end

    truth = struct();
    for fieldIndex = 1:numel(fieldNames)
        truth.(fieldNames{fieldIndex}) = nan(double(sampleCount), 1);
    end
end
