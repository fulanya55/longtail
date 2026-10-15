function paths = resolve_closedset_output_paths(cfg)
%RESOLVE_CLOSEDSET_OUTPUT_PATHS 生成闭集final与partial精确路径。
%   PATHS = RESOLVE_CLOSEDSET_OUTPUT_PATHS(CFG) 从 CFG.output_dir 和可选
%   CFG.output.dataset_stem 构造训练集、测试集及其临时文件路径。新四类、
%   五类配置显式提供 dataset_stem；旧六类配置没有该字段时，函数回退为
%   closedset_6class，从而不改变旧入口的文件名，也不改变旧MAT内保存的
%   config结构。

    validCfg = isstruct(cfg) && isscalar(cfg) && ...
        isfield(cfg, 'output_dir') && isfield(cfg, 'output') && ...
        isstruct(cfg.output) && isscalar(cfg.output);
    if ~validCfg
        error('JammingDataset:InvalidRunnerInput', ...
            'cfg必须包含output_dir和标量output结构体。');
    end

    if isfield(cfg.output, 'dataset_stem')
        stem = cfg.output.dataset_stem;
    else
        stem = 'closedset_6class';
    end
    if ~(ischar(stem) && isrow(stem) && ~isempty(stem))
        error('JammingDataset:InvalidRunnerInput', ...
            'cfg.output.dataset_stem必须是已规范化的非空字符行。');
    end

    paths = struct( ...
        'train_name', [stem '_train.mat'], ...
        'test_name', [stem '_test.mat'], ...
        'train_partial_name', [stem '_train.partial.mat'], ...
        'test_partial_name', [stem '_test.partial.mat']);
    paths.train = fullfile(cfg.output_dir, paths.train_name);
    paths.test = fullfile(cfg.output_dir, paths.test_name);
    paths.train_partial = fullfile( ...
        cfg.output_dir, paths.train_partial_name);
    paths.test_partial = fullfile( ...
        cfg.output_dir, paths.test_partial_name);
end
