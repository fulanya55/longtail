function result = generate_p3op_complex_coherence_level(executeGeneration, baseSeed, roleName, regimeIndex, powerLevel)
%GENERATE_P3OP_COMPLEX_COHERENCE_LEVEL Generate one atomic P3o-P view.
    if nargin < 1, executeGeneration = false; end
    if nargin < 2, baseSeed = 1800000000; end
    if nargin < 3, roleName = 'train'; end
    if nargin < 4, regimeIndex = 1; end
    if nargin < 5, powerLevel = 25; end
    names = {'ID_control', 'low_count', 'close_spacing', 'ultra_close', 'weak_target'};
    countRanges = {[3, 8], [1, 2], [3, 8], [3, 8], [3, 8]};
    delayRanges = {[30e-6, 50e-6], [30e-6, 50e-6], [30e-6, 34e-6], [30e-6, 31e-6], [30e-6, 50e-6]};
    amplitudeRanges = {[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.05, 1.5], [0.05, 1.5]};
    if ~(strcmp(roleName, 'train') || strcmp(roleName, 'development')), error('roleName must be train or development.'); end
    if ~(isscalar(regimeIndex) && regimeIndex >= 1 && regimeIndex <= 5 && regimeIndex == floor(regimeIndex)), error('regimeIndex must be 1..5.'); end
    if ~ismember(powerLevel, [25, 35, 50]), error('powerLevel must be 25, 35 or 50.'); end
    generatorRoot = 'C:\\LJQ_code\\干扰数据\\干扰信号生成程序\\可配置组合干扰数据生成'; outputBase = 'D:\\CJ0724_generated_data';
    if ~isfolder(generatorRoot), error('Generator root is missing: %s', generatorRoot); end
    if ~isfolder(outputBase), mkdir(outputBase); end
    probe = [tempname(outputBase), '.write_probe']; [id, message] = fopen(probe, 'w'); if id < 0, error('Output is not writable: %s', message); end; fclose(id); delete(probe);
    if executeGeneration && java.io.File(outputBase).getUsableSpace() < 5 * 1024^3, error('Need at least 5 GiB free output space.'); end
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg = create_power_sweep_6class_config(powerLevel);
    cfg.seed = baseSeed; cfg.params.MFT.num_targets = countRanges{regimeIndex}; cfg.params.MFT.target_delay_s = delayRanges{regimeIndex}; cfg.params.MFT.target_relative_amplitude = amplitudeRanges{regimeIndex};
    cfg.dataset.samples_per_pattern = 33; cfg.dataset.num_samples = 63 * 33;
    prefix = sprintf('p3op_complex_coherence_%s_seed%d', roleName, baseSeed); cfg.experiment.name = sprintf('%s_%s', prefix, names{regimeIndex});
    allLevels = 10:5:50; levelIndex = find(allLevels == powerLevel, 1, 'first'); cfg.output_dir = fullfile(outputBase, sprintf('%s_%s', prefix, names{regimeIndex}), sprintf('level_%02d__jnr_%ddB', levelIndex, powerLevel));
    result = run_power_sweep_levels({cfg}, logical(executeGeneration), 'train', 'resume', logical(executeGeneration));
    if executeGeneration, levelReceipt = result; %#ok<NASGU> save(fullfile(cfg.output_dir, 'generation_level_receipt.mat'), 'levelReceipt', '-v7.3'); end
end
