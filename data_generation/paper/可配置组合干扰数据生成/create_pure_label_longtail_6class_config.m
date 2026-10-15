function cfg = create_pure_label_longtail_6class_config(countsTrain, countsTest, outputDir, selectedTypes)
%CREATE_PURE_LABEL_LONGTAIL_6CLASS_CONFIG Configure exact-count label LT.
%   COUNTS_TRAIN/COUNTS_TEST each contain 63 entries in mask order 1:63.
%   This function starts from the audited six-class physical configuration;
%   only the combination counts and output directory are changed.

    if nargin < 1 || isempty(countsTrain)
        error('JammingConfig:PureLabelCounts', ...
            '必须显式提供 train 计数向量，避免误用默认长尾。');
    end
    if nargin < 2 || isempty(countsTest)
        countsTest = countsTrain;
    end
    if nargin < 3 || isempty(outputDir)
        outputDir = fullfile(pwd, 'generated_data', ...
            'six_class_pure_label_longtail_v1');
    end
    if nargin < 4 || isempty(selectedTypes)
        selectedTypes = {'FMB','SawFM','SinFM','MFT','VFT','ISRJ'};
    end
    if isstring(selectedTypes)
        selectedTypes = cellstr(selectedTypes);
    end
    requiredTypesAma = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    requiredTypesVft = {'FMB','SawFM','SinFM','MFT','VFT','ISRJ'};
    validTypeSet = iscell(selectedTypes) && numel(selectedTypes) == 6 && ...
        (isequal(sort(selectedTypes), sort(requiredTypesAma)) || ...
         isequal(sort(selectedTypes), sort(requiredTypesVft)));
    if ~validTypeSet
        error('JammingConfig:PureLabelTypes', ...
            'selectedTypes 必须为 AMA 版或 VFT 版六类集合。');
    end
    cfg = create_closedset_6class_config();
    cfg.output_dir = outputDir;
    cfg.types.enabled = selectedTypes;
    cfg.types.known = selectedTypes;
    cfg.types.unknown = {};
    cfg.dataset.num_samples = sum(double(countsTrain));
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 1;
    cfg.experiment.pure_label_longtail = struct( ...
        'schema', 'six-class-pure-label-longtail-v1', ...
        'labels', {selectedTypes}, ...
        'counts_train', double(reshape(countsTrain, 1, [])), ...
        'counts_test', double(reshape(countsTest, 1, [])), ...
        'min_per_pattern', 1, ...
        'anchor', 'uniform_63_patterns');
end
