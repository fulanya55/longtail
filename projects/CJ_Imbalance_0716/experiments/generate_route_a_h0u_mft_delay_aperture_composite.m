function receipt = generate_route_a_h0u_mft_delay_aperture_composite(executeGeneration)
% Fresh external H0u source: weak MFT with deliberately separated replicas.
if nargin < 1; executeGeneration = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath'))); codeRoot = fileparts(projectRoot);
hit = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
if isempty(hit); error('RouteA:GeneratorNotFound', 'Missing official generator.'); end
generatorRoot = hit(1).folder; dataRoot = fileparts(fileparts(generatorRoot));
outputRoot = fullfile(dataRoot, 'h0u_mft_delay_aperture_20260803_r1');
auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_h0u_mft_delay_aperture_generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end

baseSeed = 3900000000; pairCount = 640; backgroundJnrDb = 20; mftJnrDb = 8;
receipt = base_receipt(executeGeneration, baseSeed, pairCount, backgroundJnrDb, mftJnrDb);
if ~executeGeneration
    write_receipt(auditRoot, 'dry_run', receipt); return;
end
if isfolder(outputRoot) && ~isempty(dir(fullfile(outputRoot, '*')))
    error('RouteA:NoOverwrite', 'Formal H0u source already exists.');
end
if ~isfolder(outputRoot); mkdir(outputRoot); end

addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
cfg = create_closedset_6class_config();
cfg.radar.pri_s = 1e-3; cfg.radar.num_pulses = 1; cfg.output.save_legacy = false;
cfg.params.common.JNR_dB = [backgroundJnrDb, backgroundJnrDb];
cfg.params.MFT.num_targets = [3, 3]; cfg.params.MFT.target_delay_s = [160e-6, 700e-6];
cfg = normalize_and_validate_config(cfg, 'balanced');
registry = build_jamming_registry(cfg); schema = build_parameter_schema(registry);
reference = build_reference_signal(cfg); background = {'FMB', 'SawFM', 'SinFM', 'VFT', 'ISRJ'};
sampleLength = cfg.derived.sample_length;
absent = make_arm(sampleLength, pairCount); present = make_arm(sampleLength, pairCount);
for pairIndex = 1:pairCount
    sampleSeed = baseSeed + pairIndex - 1;
    row = make_row(2 * pairIndex - 1, sampleSeed, background, registry);
    withoutMft = compose_jamming_sample(cfg, registry, schema, row, reference);
    [mftSignal, delaySamples, amplitude, mftSeed] = separated_mft(reference.train, sampleSeed, mftJnrDb);
    withMft = withoutMft; withMft.iq_complex = withoutMft.iq_complex + mftSignal;
    absent = store(absent, pairIndex, withoutMft, sampleSeed, false, background, registry, nan(1, 3), nan(1, 3), NaN);
    present = store(present, pairIndex, withMft, sampleSeed, true, background, registry, delaySamples, amplitude, mftSeed);
end
absentPath = fullfile(outputRoot, 'mft_absent.mat'); presentPath = fullfile(outputRoot, 'mft_present_weak_separated.mat');
write_arm(absentPath, absent); write_arm(presentPath, present);
receipt.absent_path = absentPath; receipt.present_path = presentPath; receipt.sample_length = sampleLength;
write_receipt(auditRoot, 'formal', receipt);
end

function [signal, delays, amplitude, componentSeed] = separated_mft(reference, sampleSeed, targetJnrDb)
componentSeed = mod(double(sampleSeed) + 170000003, 2^32);
stream = RandStream('mt19937ar', 'Seed', componentSeed);
centres = [2000, 4000, 6000]; jitter = randi(stream, [-200, 200], 1, 3);
delays = centres + jitter; amplitude = 0.5 + rand(stream, 1, 3);
signal = complex(zeros(size(reference)));
for index = 1:3
    signal = signal + amplitude(index) * circshift(reference, [0, delays(index)]);
end
signal = signal * sqrt(10^(targetJnrDb / 10) / max(mean(abs(signal).^2), eps));
end

function arm = make_arm(sampleLength, count)
arm = struct('iq', complex(zeros(sampleLength, count)), 'pair_seed', zeros(1, count), ...
    'receiver_noise_seed', zeros(1, count), 'mft_present', zeros(1, count), ...
    'background_component_seeds', zeros(5, count), 'mft_component_seed', nan(1, count), ...
    'mft_total_jnr_db', nan(1, count), 'mft_delay_samples', nan(3, count), ...
    'mft_amplitude', nan(3, count), 'num_pulses', zeros(1, count), 'samples_per_pri', zeros(1, count));
end

function arm = store(arm, index, item, sampleSeed, mftPresent, background, registry, delaySamples, amplitude, componentSeed)
arm.iq(:, index) = item.iq_complex(:); arm.pair_seed(index) = sampleSeed;
arm.receiver_noise_seed(index) = item.receiver_noise_seed; arm.mft_present(index) = double(mftPresent);
arm.num_pulses(index) = 1; arm.samples_per_pri(index) = 10000;
for k = 1:numel(background)
    position = find(strcmp(item.active_type_codes, background{k}), 1);
    if isempty(position); error('RouteA:BackgroundMissing', 'Background component missing.'); end
    arm.background_component_seeds(k, index) = item.component_seeds(position);
end
if mftPresent
    arm.mft_component_seed(index) = componentSeed; arm.mft_total_jnr_db(index) = 8;
    arm.mft_delay_samples(:, index) = delaySamples(:); arm.mft_amplitude(:, index) = amplitude(:);
end
end

function row = make_row(sampleId, sampleSeed, background, registry)
registryCodes = {registry.code}; ids = cellfun(@(code) find(strcmp(registryCodes, code), 1), background);
[ids, order] = sort(ids); codes = background(order); offsets = zeros(1, numel(codes)); weak = false(1, numel(codes));
row = table(double(sampleId), {'train'}, double(sampleSeed), {double(ids)}, {codes}, {strjoin(codes, '+')}, ...
    double(numel(codes)), {offsets}, {weak}, 1, 1, {[false, false, false]}, false, 0, ...
    'VariableNames', {'sample_id', 'split', 'sample_seed', 'active_type_ids', 'active_type_codes', 'combination_id', ...
    'combination_size', 'component_target_power_dB', 'weak_component_mask', 'component_sampling_weight', ...
    'pattern_sampling_weight', 'imbalance_flags', 'unknown_present', 'unknown_component_count'});
end

function write_arm(path, arm)
h5create(path, '/iq_real', size(real(arm.iq)), 'Datatype', 'single'); h5write(path, '/iq_real', single(real(arm.iq)));
h5create(path, '/iq_imag', size(imag(arm.iq)), 'Datatype', 'single'); h5write(path, '/iq_imag', single(imag(arm.iq)));
names = {'pair_seed', 'receiver_noise_seed', 'mft_present', 'background_component_seeds', 'mft_component_seed', 'mft_total_jnr_db', 'mft_delay_samples', 'mft_amplitude', 'num_pulses', 'samples_per_pri'};
for k = 1:numel(names); value = arm.(names{k}); h5create(path, ['/' names{k}], size(value), 'Datatype', 'double'); h5write(path, ['/' names{k}], double(value)); end
end

function receipt = base_receipt(executed, baseSeed, pairCount, backgroundJnrDb, mftJnrDb)
receipt = struct('status', 'planned', 'executed', logical(executed), 'experiment', 'RouteA_H0u_MFTDelayAperture', ...
    'evidence_role', 'fresh_external_delay_aperture_paired_physical_identifiability', 'base_sample_seed', baseSeed, ...
    'sample_seed_range', [baseSeed, baseSeed + pairCount - 1], 'pair_count', pairCount, 'fs_hz', 1e7, ...
    'pulse_width_us', 100, 'pri_us', 1000, 'num_pulses', 1, 'background_codes', {{'FMB', 'SawFM', 'SinFM', 'VFT', 'ISRJ'}}, ...
    'background_common_jnr_db', backgroundJnrDb, 'mft_total_jnr_db', mftJnrDb, 'mft_centres_samples', [2000, 4000, 6000], ...
    'mft_jitter_samples', [-200, 200], 'mft_minimum_spacing_samples', 1600, 'test_split_generated', false, ...
    'test_files_opened', false, 'locked_test_used', false, 'test_v2_used', false, 'test_v3_used', false);
if executed; receipt.status = 'completed'; end
end

function write_receipt(root, prefix, receipt)
path = fullfile(root, [prefix '_generation_receipt.json']); fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0; error('RouteA:ReceiptWriteFailed', 'Cannot write receipt.'); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char'); fprintf('ROUTE_A_H0U_RECEIPT=%s\n', path);
end
