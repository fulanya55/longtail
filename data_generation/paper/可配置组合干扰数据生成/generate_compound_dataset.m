function result = generate_compound_dataset(cfg)
%GENERATE_COMPOUND_DATASET 生成普通组合干扰识别与参数提取数据集。
%   RESULT = GENERATE_COMPOUND_DATASET(CFG) 使用用户在 CFG 中选择的全部
%   enabled 类型、组合阶数和参数范围构建普通组合 manifest，生成复数 I/Q、
%   多标签和参数 mask，执行强一致性校验，打印统计并保存 MAT 文件。
%   RESULT 返回标准化配置、注册表、参数模式、manifest、内存 bundle、统计
%   及 data_path。省略 CFG 时使用 CREATE_DEFAULT_CONFIG 的完整十类配置。
%
%   示例：
%       cfg = create_default_config();
%       cfg.types.enabled = {'FMB','MFT','ISRJ','Traction'};
%       cfg.dataset.num_samples = 1000;
%       result = generate_compound_dataset(cfg);

    add_local_paths();
    if nargin < 1
        cfg = create_default_config();
    end
    cfg = normalize_and_validate_config(cfg, 'balanced');
    ensure_output_directory(cfg.output_dir);
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_balanced_manifest(cfg, registry, 'train');
    bundle = generate_dataset_from_manifest(cfg, registry, schema, manifest);
    validate_generated_dataset(bundle, cfg, registry, schema, 'balanced');
    statistics = print_dataset_statistics(bundle, registry);
    dataPath = fullfile(cfg.output_dir, 'compound_dataset_balanced.mat');
    save_dataset_bundle(dataPath, bundle, cfg, registry, schema);

    result = struct('scenario', 'balanced', 'config', cfg, ...
        'registry', registry, 'parameter_schema', schema, ...
        'manifest', manifest, 'bundle', bundle, ...
        'statistics', statistics, 'data_path', dataPath);
end

function add_local_paths()
%ADD_LOCAL_PATHS 使入口从任意当前工作目录调用时都能找到公共内核。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end

function ensure_output_directory(pathValue)
%ENSURE_OUTPUT_DIRECTORY 创建用户配置的输出目录。

    if ~isfolder(pathValue)
        mkdir(pathValue);
    end
end
