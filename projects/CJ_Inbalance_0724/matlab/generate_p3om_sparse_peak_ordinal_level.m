function result = generate_p3om_sparse_peak_ordinal_level(executeGeneration, baseSeed, roleName, regimeIndex, powerLevel)
%GENERATE_P3OM_SPARSE_PEAK_ORDINAL_LEVEL Generate one P3o-M view safely.
% Invoke each view in a separate MATLAB process.  This preserves the exact
% deterministic configuration of the three-level wrapper while limiting each
% HDF5 process lifetime to one atomic train transaction.
    if nargin < 1, executeGeneration = false; end
    if nargin < 2, baseSeed = 1570000000; end
    if nargin < 3, roleName = 'train'; end
    if nargin < 4, regimeIndex = 1; end
    if nargin < 5, powerLevel = 25; end
    names = {'ID_control', 'low_count', 'close_spacing', 'ultra_close', 'weak_target'};
    countRanges = {[3, 8], [1, 2], [3, 8], [3, 8], [3, 8]};
    delayRanges = {[30e-6, 50e-6], [30e-6, 50e-6], [30e-6, 34e-6], [30e-6, 31e-6], [30e-6, 50e-6]};
    amplitudeRanges = {[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.05, 1.5], [0.05, 1.5]};
    if ~(strcmp(roleName, 'train') || strcmp(roleName, 'development'))
        error('P3o-M roleName must be train or development.');
    end
    if ~(isscalar(regimeIndex) && regimeIndex >= 1 && regimeIndex <= numel(names) && regimeIndex == floor(regimeIndex))
        error('regimeIndex must be an integer from 1 through %d.', numel(names));
    end
    allLevels = 10:5:50;
    if ~ismember(powerLevel, [25, 35, 50])
        error('P3o-M powerLevel must be 25, 35, or 50 dB.');
    end
    generatorRoot = 'C:\\LJQ_code\\干扰数据\\干扰信号生成程序\\可配置组合干扰数据生成';
    outputBase = 'D:\\CJ0724_generated_data';
    validate_single_view_environment(generatorRoot, outputBase, logical(executeGeneration));
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    prefix = sprintf('p3om_sparse_peak_%s_seed%d', roleName, baseSeed);
    outputRoot = fullfile(outputBase, sprintf('%s_%s', prefix, names{regimeIndex}));
    cfg = create_power_sweep_6class_config(powerLevel);
    cfg.seed = baseSeed;
    cfg.params.MFT.num_targets = countRanges{regimeIndex};
    cfg.params.MFT.target_delay_s = delayRanges{regimeIndex};
    cfg.params.MFT.target_relative_amplitude = amplitudeRanges{regimeIndex};
    cfg.dataset.samples_per_pattern = 33;
    cfg.dataset.num_samples = 63 * cfg.dataset.samples_per_pattern;
    cfg.experiment.name = sprintf('%s_%s', prefix, names{regimeIndex});
    levelIndex = find(allLevels == powerLevel, 1, 'first');
    cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, powerLevel));
    result = run_power_sweep_levels({cfg}, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
    if executeGeneration
        levelReceipt = result; %#ok<NASGU>
        save(fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, powerLevel), ...
            'generation_level_receipt.mat'), 'levelReceipt', '-v7.3');
    end
end

function validate_single_view_environment(generatorRoot, outputBase, executeGeneration)
    if ~isfolder(generatorRoot)
        error('Generator source directory does not exist: %s', generatorRoot);
    end
    if ~isfolder(outputBase)
        [created, message] = mkdir(outputBase);
        if ~created, error('Cannot create output directory %s: %s', outputBase, message); end
    end
    probePath = [tempname(outputBase), '.write_probe'];
    [probeId, message] = fopen(probePath, 'w');
    if probeId < 0, error('Output directory is not writable (%s): %s', outputBase, message); end
    fclose(probeId); delete(probePath);
    if executeGeneration
        minFreeBytes = 5 * 1024^3;
        usableBytes = java.io.File(outputBase).getUsableSpace();
        if usableBytes < minFreeBytes
            error('Insufficient output space: %0.2f GiB available; need %0.2f GiB.', usableBytes / 1024^3, minFreeBytes / 1024^3);
        end
    end
end
