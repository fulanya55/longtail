function result = generate_closedset_6class_power_sweep( ...
        executeGeneration, levels_dB, splitSelection, existingPolicy)
%GENERATE_CLOSEDSET_6CLASS_POWER_SWEEP 生成六类九挡配对功率扫描数据。
%   RESULT = GENERATE_CLOSEDSET_6CLASS_POWER_SWEEP() 默认正式串行处理
%   10:5:50 全部九挡，每挡先 train 后 test，并以 resume 策略复用已经
%   通过增强回载的最终文件。
%
%   RESULT = GENERATE_CLOSEDSET_6CLASS_POWER_SWEEP(EXECUTEGENERATION,
%   LEVELS_DB, SPLITSELECTION, EXISTINGPOLICY) 支持：
%     EXECUTEGENERATION：true 正式生成；false 只构建和核对 manifest；
%     LEVELS_DB：10:5:50 的非空无重复子集，输入顺序即执行顺序；
%     SPLITSELECTION：'both'、'train' 或 'test'；
%     EXISTINGPOLICY：'resume'、'error' 或 'overwrite'。
%
%   九挡使用相同基础种子和组合顺序。函数在每挡执行时对完整 seed 向量
%   做 ISEQUAL 精确比较，但 RESULT 只保留轻量签名、状态、路径、统计和
%   回载结果，不保留18张大型 manifest 或任何波形 bundle。正式运行会在
%   power_sweep_9level_v1 根目录原子更新 power_sweep_manifest.csv；干运行
%   不创建目录、CSV或MAT文件。

    add_local_paths();
    if nargin < 1
        executeGeneration = true;
    end
    if nargin < 2
        levels_dB = 10:5:50;
    end
    if nargin < 3
        splitSelection = 'both';
    end
    if nargin < 4
        existingPolicy = 'resume';
    end

    executeGeneration = normalize_execute_flag(executeGeneration);
    levels_dB = normalize_levels(levels_dB);
    configs = cell(numel(levels_dB), 1);
    for levelIndex = 1:numel(levels_dB)
        configs{levelIndex} = ...
            create_power_sweep_6class_config(levels_dB(levelIndex));
    end
    result = run_power_sweep_levels(configs, executeGeneration, ...
        splitSelection, existingPolicy, executeGeneration);
end

function add_local_paths()
%ADD_LOCAL_PATHS 保证从任意工作目录调用时均可找到公共内核和生成器。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end

function value = normalize_execute_flag(value)
%NORMALIZE_EXECUTE_FLAG 规范逻辑或0/1数值执行开关。

    valid = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
        isreal(value) && isfinite(double(value)) && ...
        ismember(double(value), [0, 1]);
    if ~valid
        error('JammingPowerSweep:InvalidExecuteFlag', ...
            'executeGeneration 必须是逻辑标量或数值0/1。');
    end
    value = logical(value);
end

function levels = normalize_levels(levels)
%NORMALIZE_LEVELS 校验并规范九挡集合的非空无重复子集。

    standardLevels = 10:5:50;
    valid = isnumeric(levels) && isreal(levels) && isvector(levels) && ...
        ~isempty(levels) && all(isfinite(levels)) && ...
        all(ismember(double(levels), standardLevels));
    if valid
        levels = reshape(double(levels), 1, []);
        valid = numel(unique(levels, 'stable')) == numel(levels);
    end
    if ~valid
        error('JammingPowerSweep:InvalidLevels', ...
            'levels_dB 必须是 10:5:50 的非空无重复数值子集。');
    end
end
