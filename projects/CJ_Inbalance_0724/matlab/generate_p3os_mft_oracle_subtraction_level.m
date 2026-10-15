function result = generate_p3os_mft_oracle_subtraction_level(executeGeneration, baseSeed, regimeIndex, powerLevel, outputBase)
%GENERATE_P3OS_MFT_ORACLE_SUBTRACTION_LEVEL Generate one atomic P3o-S view.
% Run each regime-power view in a distinct MATLAB process. OUTPUTBASE defaults
% to the repaired D: root; a different root must be explicitly supplied by the
% caller after its storage authorization has been recorded.
    if nargin < 1, executeGeneration = false; end
    if nargin < 2, baseSeed = 1920000000; end
    if nargin < 3, regimeIndex = 1; end
    if nargin < 4, powerLevel = 25; end
    if nargin < 5, outputBase = 'D:\CJ0724_generated_data'; end
    names = {'ID_control', 'low_count', 'close_spacing', 'ultra_close', 'weak_target'};
    countRanges = {[3, 8], [1, 2], [3, 8], [3, 8], [3, 8]};
    delayRanges = {[30e-6, 50e-6], [30e-6, 50e-6], [30e-6, 34e-6], [30e-6, 31e-6], [30e-6, 50e-6]};
    amplitudeRanges = {[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.05, 1.5]};
    allLevels = 10:5:50;
    if ~(isscalar(baseSeed) && isfinite(baseSeed) && baseSeed == floor(baseSeed)), error('baseSeed must be a finite integer.'); end
    if ~(isscalar(regimeIndex) && regimeIndex >= 1 && regimeIndex <= 5 && regimeIndex == floor(regimeIndex)), error('regimeIndex must be 1..5.'); end
    if ~ismember(powerLevel, [25, 35, 50]), error('powerLevel must be 25, 35 or 50.'); end
    if ~(ischar(outputBase) || isstring(outputBase)), error('outputBase must be a path.'); end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputBase = char(outputBase);
    if ~isfolder(generatorRoot), error('Generator root is missing: %s', generatorRoot); end
    if ~isfolder(outputBase), mkdir(outputBase); end
    probe = [tempname(outputBase), '.write_probe']; [id, message] = fopen(probe, 'w'); if id < 0, error('Output is not writable: %s', message); end; fclose(id); delete(probe);
    if executeGeneration && java.io.File(outputBase).getUsableSpace() < 5 * 1024^3, error('Need at least 5 GiB free output space.'); end
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg = create_power_sweep_6class_config(powerLevel); cfg.seed = baseSeed;
    cfg.params.MFT.num_targets = countRanges{regimeIndex}; cfg.params.MFT.target_delay_s = delayRanges{regimeIndex}; cfg.params.MFT.target_relative_amplitude = amplitudeRanges{regimeIndex};
    cfg.dataset.samples_per_pattern = 33; cfg.dataset.num_samples = 63 * 33;
    prefix = sprintf('p3os_mft_oracle_subtraction_seed%d', baseSeed);
    cfg.experiment.name = sprintf('%s_%s', prefix, names{regimeIndex}); levelIndex = find(allLevels == powerLevel, 1, 'first');
    cfg.output_dir = fullfile(outputBase, sprintf('%s_%s', prefix, names{regimeIndex}), sprintf('level_%02d__jnr_%ddB', levelIndex, powerLevel));
    result = run_power_sweep_levels({cfg}, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
    if executeGeneration, levelReceipt = result; %#ok<NASGU> save(fullfile(cfg.output_dir, 'generation_level_receipt.mat'), 'levelReceipt', '-v7.3'); end
end
