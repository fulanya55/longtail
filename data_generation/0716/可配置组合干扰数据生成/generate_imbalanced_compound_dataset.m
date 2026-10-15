function result = generate_imbalanced_compound_dataset(cfg)
%GENERATE_IMBALANCED_COMPOUND_DATASET 生成多轴不均衡组合干扰数据集。
%   RESULT = GENERATE_IMBALANCED_COMPOUND_DATASET(CFG) 支持独立或联合启用：
%   1) component.weights 控制成分边际不均衡；
%   2) cardinality_weights/pattern_weights 控制组合阶数或模式不均衡；
%   3) relative_power_dB 与 weak_component_policy 控制功率弱分量不均衡。
%   功率偏置会真实施加到波形，并保存最终绝对/相对功率和有效参数真值。
%   省略 CFG 时使用默认配置；默认不均衡开关均关闭，用户可按需开启。
%
%   示例：
%       cfg = create_default_config();
%       cfg.imbalance.component.enable = true;
%       cfg.imbalance.component.weights = [10 5 2 1 1 1 1 1 1 1];
%       cfg.imbalance.power.enable = true;
%       cfg.imbalance.power.relative_power_dB = [-20 -8];
%       result = generate_imbalanced_compound_dataset(cfg);

    add_local_paths();
    if nargin < 1
        cfg = create_default_config();
    end
    cfg = normalize_and_validate_config(cfg, 'imbalanced');
    ensure_output_directory(cfg.output_dir);
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_imbalanced_manifest(cfg, registry, 'train');
    bundle = generate_dataset_from_manifest(cfg, registry, schema, manifest);
    validate_generated_dataset(bundle, cfg, registry, schema, 'imbalanced');
    statistics = print_dataset_statistics(bundle, registry);
    dataPath = fullfile(cfg.output_dir, 'compound_dataset_imbalanced.mat');
    save_dataset_bundle(dataPath, bundle, cfg, registry, schema);

    result = struct('scenario', 'imbalanced', 'config', cfg, ...
        'registry', registry, 'parameter_schema', schema, ...
        'manifest', manifest, 'bundle', bundle, ...
        'statistics', statistics, 'data_path', dataPath);
end

function add_local_paths()
%ADD_LOCAL_PATHS 注册入口目录、公共内核和十类生成器目录。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end

function ensure_output_directory(pathValue)
%ENSURE_OUTPUT_DIRECTORY 创建用户配置的输出目录。

    if ~isfolder(pathValue)
        mkdir(pathValue);
    end
end
