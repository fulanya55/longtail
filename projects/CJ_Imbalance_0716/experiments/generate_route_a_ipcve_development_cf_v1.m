function receipt = generate_route_a_ipcve_development_cf_v1(mode)
% Generate identity-matched absent/low/high triples for Route-A IP-CVE.
% MODE is dry_run, smoke, or formal. No test or confirmation source is made.

if nargin < 1
    mode = 'dry_run';
end
mode = char(string(mode));
if ~ismember(mode, {'dry_run','smoke','formal'})
    error('RouteAIPCV:InvalidMode', 'mode must be dry_run, smoke, or formal');
end

projectRoot = 'E:\LJQ_code\CJ_Imbalance_0716';
generatorRoot = ['E:\LJQ_code\干扰数据\干扰信号生成程序\' ...
    '可配置组合干扰数据生成'];
outputRoot = ['E:\LJQ_code\干扰数据\' ...
    'route_a_ipcve_development_cf_v1_20260807'];
auditRoot = fullfile(projectRoot, 'outputs', 'route_a_ipcve', ...
    'stage0', 'generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end
if ~isfolder(generatorRoot)
    error('RouteAIPCV:GeneratorNotFound', 'Generator root is missing: %s', generatorRoot);
end

switch mode
    case 'dry_run'
        baseSeed = 4000000000;
        tripleCount = 2400;
        dataPath = fullfile(outputRoot, 'ipcve_triples_train.mat');
    case 'smoke'
        baseSeed = 3990000000;
        tripleCount = 15;
        dataPath = fullfile(outputRoot, 'ipcve_triples_smoke.mat');
    otherwise
        baseSeed = 4000000000;
        tripleCount = 2400;
        dataPath = fullfile(outputRoot, 'ipcve_triples_train.mat');
end

addpath(generatorRoot, fullfile(generatorRoot, 'core'), ...
    fullfile(generatorRoot, 'generators'));
cfg = create_closedset_6class_config();
cfg.seed = baseSeed;
cfg.output.save_legacy = false;
cfg.params.common.JNR_dB = [20,20];
cfg = normalize_and_validate_config(cfg, 'balanced');
registry = build_jamming_registry(cfg);
schema = build_parameter_schema(registry);
reference = build_reference_signal(cfg);

expectedReceiverRange = [baseSeed + 198677742, ...
    baseSeed + 198677742 + tripleCount - 1];
receipt = base_receipt(mode, false, dataPath, baseSeed, tripleCount, ...
    expectedReceiverRange);
if strcmp(mode, 'dry_run')
    receipt.status = 'completed_dry_run_no_data_generated';
    write_receipt(auditRoot, mode, receipt);
    disp(jsonencode(receipt, PrettyPrint=true));
    return;
end

if isfile(dataPath)
    error('RouteAIPCV:NoOverwrite', ...
        'IP-CVE source already exists and must not be overwritten: %s', dataPath);
end
if ~isfolder(outputRoot); mkdir(outputRoot); end

targets = {'MFT','VFT','ISRJ'};
nuisance = cell(3,5);
nuisance(1,:) = {{'FMB','SawFM'}, {'FMB','VFT'}, ...
    {'SawFM','ISRJ'}, {'VFT','ISRJ'}, {'FMB','SawFM','VFT'}};
nuisance(2,:) = {{'FMB','SawFM'}, {'FMB','MFT'}, ...
    {'SawFM','ISRJ'}, {'MFT','ISRJ'}, {'FMB','SawFM','MFT'}};
nuisance(3,:) = {{'FMB','SawFM'}, {'FMB','VFT'}, ...
    {'SawFM','VFT'}, {'MFT','VFT'}, {'FMB','SawFM','VFT'}};

viewCount = 3 * tripleCount;
sampleLength = cfg.derived.sample_length;
parameterCount = height(schema);
iq_complex = complex(zeros(sampleLength, viewCount));
label_selected = zeros(numel(cfg.types.enabled), viewCount, 'uint8');
parameter_values = nan(parameterCount, viewCount);
parameter_mask = false(parameterCount, viewCount);
receiver_noise_seed = zeros(1, viewCount, 'uint32');
sample_seed = zeros(1, viewCount, 'uint32');
view_sample_id = zeros(1, viewCount, 'uint32');
triple_id = zeros(1, viewCount, 'uint32');
triple_target_index = zeros(1, viewCount, 'uint8');
triple_level = zeros(1, viewCount, 'uint8');
nuisance_index = zeros(1, viewCount, 'uint8');
role_code = zeros(1, viewCount, 'uint8');
target_offset_dB = nan(1, viewCount);

for tripleIndex = 1:tripleCount
    targetPosition = mod(tripleIndex - 1, 3) + 1;
    nuisancePosition = mod(floor((tripleIndex - 1) / 3), 5) + 1;
    targetCode = targets{targetPosition};
    combo = nuisance{targetPosition, nuisancePosition};
    seed = baseSeed + tripleIndex - 1;
    firstSampleId = 3 * tripleIndex - 2;
    rows = {
        make_row(firstSampleId, seed, combo, targetCode, 'absent', registry), ...
        make_row(firstSampleId + 1, seed, combo, targetCode, 'low', registry), ...
        make_row(firstSampleId + 2, seed, combo, targetCode, 'high', registry)};
    samples = {
        compose_jamming_sample(cfg, registry, schema, rows{1}, reference), ...
        compose_jamming_sample(cfg, registry, schema, rows{2}, reference), ...
        compose_jamming_sample(cfg, registry, schema, rows{3}, reference)};
    observedReceiver = cellfun(@(item) double(item.receiver_noise_seed), samples);
    if numel(unique(observedReceiver)) ~= 1
        error('RouteAIPCV:TripleNoiseMismatch', ...
            'Triple views must share receiver noise at triple %d', tripleIndex);
    end
    if observedReceiver(1) < 0 || observedReceiver(1) > double(intmax('uint32'))
        error('RouteAIPCV:ReceiverSeedOverflow', 'receiver seed exceeds uint32');
    end
    role = receiver_role(observedReceiver(1));
    columns = 3 * tripleIndex - 2 : 3 * tripleIndex;
    for localIndex = 1:3
        column = columns(localIndex);
        item = samples{localIndex};
        iq_complex(:, column) = item.iq_complex(:);
        label_selected(:, column) = uint8(item.label_selected(:));
        parameter_values(:, column) = item.parameter_values(:);
        parameter_mask(:, column) = logical(item.parameter_mask(:));
        receiver_noise_seed(column) = uint32(item.receiver_noise_seed);
        sample_seed(column) = uint32(item.sample_seed);
        view_sample_id(column) = uint32(firstSampleId + localIndex - 1);
        triple_id(column) = uint32(tripleIndex);
        triple_target_index(column) = uint8(find( ...
            strcmp(cfg.types.enabled, targetCode), 1));
        triple_level(column) = uint8(localIndex - 1);
        nuisance_index(column) = uint8(nuisancePosition);
        role_code(column) = uint8(role);
        if localIndex == 2; target_offset_dB(column) = -12; end
        if localIndex == 3; target_offset_dB(column) = 0; end
    end
end

save(dataPath, 'iq_complex', 'label_selected', 'parameter_values', ...
    'parameter_mask', 'receiver_noise_seed', 'sample_seed', ...
    'view_sample_id', 'triple_id', 'triple_target_index', ...
    'triple_level', 'nuisance_index', 'role_code', ...
    'target_offset_dB', '-v7.3');

receipt = base_receipt(mode, true, dataPath, baseSeed, tripleCount, ...
    expectedReceiverRange);
receipt.status = ['completed_' mode '_train_only_generation'];
receipt.observed_receiver_noise_seed_range = [ ...
    double(min(receiver_noise_seed)), double(max(receiver_noise_seed))];
receipt.role_triple_counts = [ ...
    sum(role_code(1:3:end) == 0), ...
    sum(role_code(1:3:end) == 1), ...
    sum(role_code(1:3:end) == 2)];
write_receipt(auditRoot, mode, receipt);
disp(jsonencode(receipt, PrettyPrint=true));
end

function row = make_row(sampleId, sampleSeed, nuisance, targetCode, level, registry)
codes = nuisance;
if ~strcmp(level, 'absent'); codes = [codes, {targetCode}]; end
registryCodes = {registry.code};
ids = cellfun(@(code) find(strcmp(registryCodes, code), 1), codes);
[ids, order] = sort(ids); codes = codes(order);
offsets = zeros(1, numel(codes));
weak = false(1, numel(codes));
if ~strcmp(level, 'absent')
    targetMask = strcmp(codes, targetCode);
    if strcmp(level, 'low')
        offsets(targetMask) = -12;
        weak(targetMask) = true;
    elseif strcmp(level, 'high')
        offsets(targetMask) = 0;
    else
        error('RouteAIPCV:InvalidLevel', 'Unknown level: %s', level);
    end
end
row = table(double(sampleId), {'train'}, double(sampleSeed), {double(ids)}, ...
    {codes}, {strjoin(codes, '+')}, double(numel(codes)), {offsets}, {weak}, ...
    1, 1, {[false,false,false]}, false, 0, 'VariableNames', ...
    {'sample_id','split','sample_seed','active_type_ids','active_type_codes', ...
    'combination_id','combination_size','component_target_power_dB', ...
    'weak_component_mask','component_sampling_weight','pattern_sampling_weight', ...
    'imbalance_flags','unknown_present','unknown_component_count'});
end

function role = receiver_role(receiverSeed)
residue = mod(double(receiverSeed), 10);
if residue <= 5
    role = 0;
elseif residue <= 7
    role = 1;
else
    role = 2;
end
end

function receipt = base_receipt(mode, executed, dataPath, baseSeed, tripleCount, range)
receipt = struct( ...
    'status', 'initialized', ...
    'experiment', 'RouteA_IPCVE_DevelopmentCFV1', ...
    'mode', mode, ...
    'executed', logical(executed), ...
    'evidence_role', 'fresh_train_only_counterfactual_triple_protocol', ...
    'base_seed', baseSeed, ...
    'triple_count', tripleCount, ...
    'view_count', 3 * tripleCount, ...
    'targets', {{'MFT','VFT','ISRJ'}}, ...
    'levels', {{'absent','low','high'}}, ...
    'common_JNR_dB', [20,20], ...
    'target_offsets_dB', [NaN,-12,0], ...
    'expected_receiver_noise_seed_range', range, ...
    'data_path', dataPath, ...
    'test_split_generated', false, ...
    'test_files_opened', false, ...
    'locked_test_used', false, ...
    'test_v2_used', false, ...
    'test_v3_used', false, ...
    'confirmation_generated', false);
end

function write_receipt(root, prefix, receipt)
pathValue = fullfile(root, [prefix '_generation_receipt.json']);
handle = fopen(pathValue, 'w', 'n', 'UTF-8');
if handle < 0
    error('RouteAIPCV:ReceiptWriteFailed', 'Cannot write receipt: %s', pathValue);
end
cleaner = onCleanup(@() fclose(handle)); %#ok<NASGU>
fprintf(handle, '%s\n', jsonencode(receipt, PrettyPrint=true));
fprintf('ROUTE_A_IPCVE_GENERATION_RECEIPT=%s\n', pathValue);
end
