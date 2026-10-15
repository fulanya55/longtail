function result = generate_closedset_6class_train_test( ...
        executeGeneration, splitSelection, existingPolicy)
%GENERATE_CLOSEDSET_6CLASS_TRAIN_TEST 原子生成 0716 六类闭集训练/测试集。
%   RESULT = GENERATE_CLOSEDSET_6CLASS_TRAIN_TEST() 默认等价于：
%       generate_closedset_6class_train_test(true, 'both', 'resume')
%   即按 train 后 test 的顺序处理两份数据；已有且通过完整回载验证的
%   最终文件直接复用，适合中断后再次执行。
%
%   RESULT = GENERATE_CLOSEDSET_6CLASS_TRAIN_TEST(EXECUTEGENERATION,
%   SPLITSELECTION, EXISTINGPOLICY) 支持：
%     EXECUTEGENERATION：true 正式生成；false 只构建 manifest，不写文件。
%     SPLITSELECTION：'both'、'train' 或 'test'。
%     EXISTINGPOLICY：
%       'resume'    验证并复用匹配的最终文件；不匹配立即报错；
%       'error'     只要最终文件存在就拒绝执行；
%       'overwrite' 显式允许以本次新结果替换最终文件。
%
%   正式生成不会直接写最终文件，而是先保存同目录精确 partial 路径：
%       closedset_6class_train.partial.mat
%       closedset_6class_test.partial.mat
%   保存后先释放大型 bundle，再增强回载 partial；通过后才 MOVEFILE 到
%   最终路径并二次回载。中断最多留下 partial，不留下伪完整 final。
%
%   干运行 RESULT 返回所选划分的完整 manifest；正式运行只返回配置、
%   路径、统计和 RESULT.splits.(split).verification，不持有大型 bundle
%   或正式 manifest。训练和测试各为 25200 条、63 个模式、每模式400条。

    add_local_paths();
    if nargin < 1
        executeGeneration = true;
    end
    if nargin < 2
        splitSelection = 'both';
    end
    if nargin < 3
        existingPolicy = 'resume';
    end

    cfg = create_closedset_6class_config();
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, ...
        executeGeneration, splitSelection, existingPolicy);
end

function add_local_paths()
%ADD_LOCAL_PATHS 保证从任意当前目录调用时都能找到公共内核和生成器。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end
