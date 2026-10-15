function value = current_scientific_semantics_version()
%CURRENT_SCIENTIFIC_SEMANTICS_VERSION 返回当前数据科学语义版本。
%   该版本独立于参数注册表 v1/v2。注册表版本描述类别数量和参数列宽，
%   本字符串描述物理真值是否采用：FMB/AMA 的实际离散带宽、MFT/Range
%   的最终总时延，以及 Traction 的单一幅度尺度。
%
%   缺失本字段的历史 MAT 文件只能视为 legacy/unknown，不得自动解释为
%   已采用当前修正语义。

    value = 'observable-truth-2026-07-27';
end
