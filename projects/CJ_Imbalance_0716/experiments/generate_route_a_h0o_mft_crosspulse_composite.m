function receipt = generate_route_a_h0o_mft_crosspulse_composite(executeGeneration)
% Official-generator, paired four-PRI composite MFT attribution screen.
if nargin < 1; executeGeneration = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
codeRoot = fileparts(projectRoot);
hit = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
if isempty(hit); error('RouteA:GeneratorNotFound', 'Missing official generator.'); end
generatorRoot = hit(1).folder;
dataRoot = fileparts(fileparts(generatorRoot));
outputRoot = fullfile(dataRoot, 'h0o_mft_crosspulse_composite_20260803');
auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_h0o_mft_crosspulse_generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end

baseSeed = 2000000000; pairCount = 640; targetOffsetDb = -12;
receipt = base_receipt(executeGeneration, baseSeed, pairCount, targetOffsetDb);
if ~executeGeneration
    write_receipt(auditRoot, 'dry_run', receipt); return;
end
if isfolder(outputRoot) && ~isempty(dir(fullfile(outputRoot, '*')))
    error('RouteA:NoOverwrite', 'Formal H0o source already exists.');
end
if ~isfolder(outputRoot); mkdir(outputRoot); end
addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
cfg = create_closedset_6class_config();
cfg.radar.num_pulses = 4;
cfg.output.save_legacy = false;
cfg.params.common.JNR_dB = [20, 20];
cfg = normalize_and_validate_config(cfg, 'balanced');
registry = build_jamming_registry(cfg); schema = build_parameter_schema(registry);
reference = build_reference_signal(cfg);
background = {'FMB', 'SawFM', 'SinFM', 'VFT', 'ISRJ'};
sampleLength = cfg.derived.sample_length;
absent = make_arm(sampleLength, pairCount); present = make_arm(sampleLength, pairCount);
for pairIndex = 1:pairCount
    sampleSeed = baseSeed + pairIndex - 1;
    absentRow = make_row(2 * pairIndex - 1, sampleSeed, background, {}, registry, targetOffsetDb);
    presentRow = make_row(2 * pairIndex, sampleSeed, background, {'MFT'}, registry, targetOffsetDb);
    withoutMft = compose_jamming_sample(cfg, registry, schema, absentRow, reference);
    withMft = compose_jamming_sample(cfg, registry, schema, presentRow, reference);
    if withoutMft.receiver_noise_seed ~= withMft.receiver_noise_seed
        error('RouteA:PairNoiseMismatch', 'counterfactual pair has different receiver noise.');
    end
    absent = store(absent, pairIndex, withoutMft, sampleSeed, false, background, registry);
    present = store(present, pairIndex, withMft, sampleSeed, true, background, registry);
end
absentPath = fullfile(outputRoot, 'mft_absent.mat');
presentPath = fullfile(outputRoot, 'mft_present_weak_m12db.mat');
write_arm(absentPath, absent); write_arm(presentPath, present);
receipt.absent_path = absentPath; receipt.present_path = presentPath;
write_receipt(auditRoot, 'formal', receipt);
end

function arm = make_arm(sampleLength, count)
arm = struct('iq', complex(zeros(sampleLength, count)), 'pair_seed', zeros(1, count), ...
    'receiver_noise_seed', zeros(1, count), 'mft_present', zeros(1, count), ...
    'background_component_seeds', zeros(5, count), 'mft_component_seed', nan(1, count), ...
    'mft_effective_jnr_db', nan(1, count), 'num_pulses', zeros(1, count));
end

function arm = store(arm, index, item, sampleSeed, mftPresent, background, registry)
arm.iq(:, index) = item.iq_complex(:); arm.pair_seed(index) = sampleSeed;
arm.receiver_noise_seed(index) = item.receiver_noise_seed; arm.mft_present(index) = double(mftPresent);
arm.num_pulses(index) = 4;
for k = 1:numel(background)
    position = find(strcmp(item.active_type_codes, background{k}), 1);
    if isempty(position); error('RouteA:BackgroundMissing', 'Background component missing.'); end
    arm.background_component_seeds(k, index) = item.component_seeds(position);
end
if mftPresent
    position = find(strcmp(item.active_type_codes, 'MFT'), 1);
    arm.mft_component_seed(index) = item.component_seeds(position);
    arm.mft_effective_jnr_db(index) = item.component_effective_JNR_dB(position);
end
end

function row = make_row(sampleId, sampleSeed, background, target, registry, targetOffsetDb)
codes = [background, target]; registryCodes = {registry.code};
ids = cellfun(@(code) find(strcmp(registryCodes, code), 1), codes);
[ids, order] = sort(ids); codes = codes(order); offsets = zeros(1, numel(codes)); weak = false(1, numel(codes));
if ~isempty(target)
    mask = strcmp(codes, target{1}); offsets(mask) = targetOffsetDb; weak(mask) = true;
end
row = table(double(sampleId), {'train'}, double(sampleSeed), {double(ids)}, {codes}, ...
    {strjoin(codes, '+')}, double(numel(codes)), {offsets}, {weak}, 1, 1, ...
    {[false, false, false]}, false, 0, 'VariableNames', {'sample_id', 'split', ...
    'sample_seed', 'active_type_ids', 'active_type_codes', 'combination_id', ...
    'combination_size', 'component_target_power_dB', 'weak_component_mask', ...
    'component_sampling_weight', 'pattern_sampling_weight', 'imbalance_flags', ...
    'unknown_present', 'unknown_component_count'});
end

function write_arm(path, arm)
h5create(path, '/iq_real', size(real(arm.iq)), 'Datatype', 'double'); h5write(path, '/iq_real', real(arm.iq));
h5create(path, '/iq_imag', size(imag(arm.iq)), 'Datatype', 'double'); h5write(path, '/iq_imag', imag(arm.iq));
names = {'pair_seed', 'receiver_noise_seed', 'mft_present', 'background_component_seeds', 'mft_component_seed', 'mft_effective_jnr_db', 'num_pulses'};
for k = 1:numel(names)
    value = arm.(names{k}); h5create(path, ['/' names{k}], size(value), 'Datatype', 'double'); h5write(path, ['/' names{k}], double(value));
end
end

function receipt = base_receipt(executed, baseSeed, pairCount, targetOffsetDb)
receipt = struct('status', 'planned', 'executed', logical(executed), ...
    'experiment', 'RouteA_H0o_MFTCrossPulseCompositeAttribution', ...
    'evidence_role', 'fresh_official_generator_paired_composite_physical_identifiability', ...
    'base_sample_seed', baseSeed, 'sample_seed_range', [baseSeed, baseSeed + pairCount - 1], ...
    'pair_count', pairCount, 'num_pulses', 4, 'samples_per_pri', 3000, ...
    'background_codes', {{'FMB', 'SawFM', 'SinFM', 'VFT', 'ISRJ'}}, ...
    'background_common_JNR_dB', [20, 20], 'mft_offset_dB', targetOffsetDb, ...
    'mft_effective_JNR_dB', [8, 8], 'test_split_generated', false, ...
    'test_files_opened', false, 'locked_test_used', false, 'test_v2_used', false, 'test_v3_used', false);
if executed; receipt.status = 'completed'; end
end

function write_receipt(root, prefix, receipt)
path = fullfile(root, [prefix '_generation_receipt.json']); fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0; error('RouteA:ReceiptWriteFailed', 'Cannot write receipt.'); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(receipt, PrettyPrint=true), 'char'); fprintf('ROUTE_A_H0O_RECEIPT=%s\n', path);
end
