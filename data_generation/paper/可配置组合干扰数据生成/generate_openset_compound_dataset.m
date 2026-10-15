function result = generate_openset_compound_dataset(cfg)
%GENERATE_OPENSET_COMPOUND_DATASET 生成开放集组合干扰协议的四个数据划分。
%   RESULT = GENERATE_OPENSET_COMPOUND_DATASET(CFG) 使用用户可修改的
%   cfg.types.known 和 cfg.types.unknown 构建 train_known、
%   calibration_known、validation_mixed、test_mixed。训练/校准仅含已知类；
%   验证/测试按 mixed_ratios 包含纯已知、纯未知、已知+一个未知、已知+
%   多个未知。每个划分独立生成、校验、统计并保存为一个 MAT 文件。
%
%   RESULT.data_paths、RESULT.manifests、RESULT.bundles 和 RESULT.statistics
%   均使用上述四个稳定字段名。未知成分仍保存完整 label_all 和参数真值，
%   但训练算法应只使用 train_known 中的 known_label，避免未知身份泄露。
%
%   示例：
%       cfg = create_default_config();
%       cfg.types.known = {'FMB','AMA','MFT','ISRJ'};
%       cfg.types.unknown = {'SawFM','SinFM','SN','VFT', ...
%           'RangeFakeTarget','Traction'};
%       cfg.types.enabled = [cfg.types.known cfg.types.unknown];
%       result = generate_openset_compound_dataset(cfg);

    add_local_paths();
    if nargin < 1
        cfg = create_default_config();
    end
    cfg = normalize_and_validate_config(cfg, 'openset');
    ensure_output_directory(cfg.output_dir);
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifests = build_openset_manifest(cfg, registry);
    splitFields = {'train_known', 'calibration_known', ...
        'validation_mixed', 'test_mixed'};
    fileNames = {'openset_train_known.mat', ...
        'openset_calibration_known.mat', ...
        'openset_validation_mixed.mat', 'openset_test_mixed.mat'};
    bundles = struct();
    statistics = struct();
    dataPaths = struct();

    for splitIndex = 1:numel(splitFields)
        fieldName = splitFields{splitIndex};
        manifest = manifests.(fieldName);
        bundle = generate_dataset_from_manifest( ...
            cfg, registry, schema, manifest);
        validate_generated_dataset( ...
            bundle, cfg, registry, schema, 'openset');
        statistics.(fieldName) = print_dataset_statistics(bundle, registry);
        dataPaths.(fieldName) = fullfile(cfg.output_dir, ...
            fileNames{splitIndex});
        save_dataset_bundle(dataPaths.(fieldName), ...
            bundle, cfg, registry, schema);
        bundles.(fieldName) = bundle;
    end

    result = struct('scenario', 'openset', 'config', cfg, ...
        'registry', registry, 'parameter_schema', schema, ...
        'manifests', manifests, 'bundles', bundles, ...
        'statistics', statistics, 'data_paths', dataPaths);
end

function add_local_paths()
%ADD_LOCAL_PATHS 注册入口目录、公共内核和各版本生成器目录。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end

function ensure_output_directory(pathValue)
%ENSURE_OUTPUT_DIRECTORY 创建用户配置的输出目录。

    if ~isfolder(pathValue)
        mkdir(pathValue);
    end
end
