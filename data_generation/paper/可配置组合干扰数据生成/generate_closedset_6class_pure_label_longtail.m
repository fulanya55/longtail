function result = generate_closedset_6class_pure_label_longtail( ...
        countsTrain, countsTest, executeGeneration, splitSelection, ...
        existingPolicy, outputDir, selectedTypes)
%GENERATE_CLOSEDSET_6CLASS_PURE_LABEL_LONGTAIL Generate exact-count LT data.
%   The first two inputs are 63-entry integer vectors in pattern-mask order.
%   The runner reuses the standard six-class physical generators and atomic
%   partial/final save path.  Existing balanced and power-sweep entry points
%   are not modified by this wrapper.

    add_local_paths();
    if nargin < 3, executeGeneration = true; end
    if nargin < 4, splitSelection = 'both'; end
    if nargin < 5, existingPolicy = 'resume'; end
    if nargin < 6, outputDir = []; end
    if nargin < 7, selectedTypes = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'}; end
    cfg = create_pure_label_longtail_6class_config( ...
        countsTrain, countsTest, outputDir, selectedTypes);
    cfg.experiment.pure_label_longtail.min_per_pattern = ...
        min([countsTrain(countsTrain > 0), countsTest(countsTest > 0)]);
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg, registry, schema, ...
        executeGeneration, splitSelection, existingPolicy);
    result.scenario = 'pure_label_longtail';
end

function add_local_paths()
    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end
