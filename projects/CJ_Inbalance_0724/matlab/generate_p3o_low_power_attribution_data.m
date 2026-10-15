function results = generate_p3o_low_power_attribution_data(executeGeneration, baseSeed, outputPrefix, samplesPerPattern, outputBase)
%GENERATE_P3O_LOW_POWER_ATTRIBUTION_DATA Generate paired 33-identity P3o data.
    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        baseSeed = 830000000;
    end
    if nargin < 3
        outputPrefix = sprintf('p3o_low_power_attribution_seed%d', baseSeed);
    end
    if nargin < 4
        samplesPerPattern = 33;
    end
    if nargin < 5
        % Keep generation artifacts off the system drive.  A full P3o role
        % contains 15 HDF5 views and can fail late in HDF5 shutdown when the
        % output volume becomes full.
        outputBase = 'D:\\CJ0724_generated_data';
    end
    if samplesPerPattern ~= 33
        error('P3o contract requires exactly 33 identities per combination.');
    end
    generatorRoot = 'C:\\LJQ_code\\干扰数据\\干扰信号生成程序\\可配置组合干扰数据生成';
    validate_generation_environment(generatorRoot, outputBase, logical(executeGeneration));
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    names = {'ID_control', 'low_count', 'close_spacing', 'ultra_close', 'weak_target'};
    countRanges = {[3, 8], [1, 2], [3, 8], [3, 8], [3, 8]};
    delayRanges = {[30e-6, 50e-6], [30e-6, 50e-6], [30e-6, 34e-6], [30e-6, 31e-6], [30e-6, 50e-6]};
    amplitudeRanges = {[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.05, 1.5]};
    levels = [25, 35, 50];
    allLevels = 10:5:50;
    results = cell(numel(names), 1);
    for regimeIndex = 1:numel(names)
        name = names{regimeIndex};
        outputRoot = fullfile(outputBase, sprintf('%s_%s', outputPrefix, name));
        configs = cell(numel(levels), 1);
        for index = 1:numel(levels)
            cfg = create_power_sweep_6class_config(levels(index));
            cfg.seed = baseSeed;
            cfg.params.MFT.num_targets = countRanges{regimeIndex};
            cfg.params.MFT.target_delay_s = delayRanges{regimeIndex};
            cfg.params.MFT.target_relative_amplitude = amplitudeRanges{regimeIndex};
            cfg.dataset.samples_per_pattern = samplesPerPattern;
            cfg.dataset.num_samples = 63 * samplesPerPattern;
            cfg.experiment.name = sprintf('%s_%s', outputPrefix, name);
            levelIndex = find(allLevels == levels(index), 1, 'first');
            cfg.output_dir = fullfile(outputRoot, sprintf('level_%02d__jnr_%ddB', levelIndex, levels(index)));
            configs{index} = cfg;
        end
        results{regimeIndex} = run_power_sweep_levels(configs, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
        if executeGeneration
            result = results{regimeIndex}; %#ok<NASGU>
            countRange = countRanges{regimeIndex}; %#ok<NASGU>
            delayRange_s = delayRanges{regimeIndex}; %#ok<NASGU>
            amplitudeRange = amplitudeRanges{regimeIndex}; %#ok<NASGU>
            save(fullfile(outputRoot, 'generation_receipt.mat'), 'result', 'countRange', 'delayRange_s', 'amplitudeRange', 'samplesPerPattern', '-v7.3');
        end
    end
end

function validate_generation_environment(generatorRoot, outputBase, executeGeneration)
%VALIDATE_GENERATION_ENVIRONMENT Fail before HDF5 creation on bad output I/O.
% A false preflight verifies source/output paths without creating waveform data.
    if ~(ischar(outputBase) || isstring(outputBase))
        error('outputBase must be a character vector or string scalar.');
    end
    outputBase = char(outputBase);
    if ~isfolder(generatorRoot)
        error('Generator source directory does not exist: %s', generatorRoot);
    end
    if ~isfolder(outputBase)
        [created, message] = mkdir(outputBase);
        if ~created
            error('Cannot create output directory %s: %s', outputBase, message);
        end
    end

    % Probe writability before the generator starts a v7.3/HDF5 transaction.
    probePath = [tempname(outputBase), '.write_probe'];
    [probeId, message] = fopen(probePath, 'w');
    if probeId < 0
        error('Output directory is not writable (%s): %s', outputBase, message);
    end
    fclose(probeId);
    delete(probePath);

    if executeGeneration
        minFreeBytes = 15 * 1024^3;
        outputVolume = java.io.File(outputBase);
        usableBytes = outputVolume.getUsableSpace();
        if usableBytes < minFreeBytes
            error(['Insufficient free space on the output volume (%0.2f GiB; ', ...
                   'need at least %0.2f GiB). Choose a volume with capacity ', ...
                   'before creating HDF5 output.'], usableBytes / 1024^3, minFreeBytes / 1024^3);
        end
    end
end
