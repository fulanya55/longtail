function receipt = generate_route_a_h1a_vft_staggered_pri_mft_isrj(executeGeneration)
% Fresh paired H1a source: weak VFT present/absent in matched MFT+ISRJ backgrounds.
% This is a project-local staggered-PRI physics extension, not official 0716 data.
if nargin < 1; executeGeneration = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath'))); codeRoot = fileparts(projectRoot);
dataRoot = fullfile(codeRoot, '干扰数据');
outputRoot = fullfile(dataRoot, 'h1a_vft_staggered_pri_mft_isrj_20260803');
auditRoot = fullfile(projectRoot, 'outputs', 'tapc', 'route_a_h1a_vft_staggered_pri_generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end

baseSeed = 2700000000; count = 640; fs = 10e6; pulseSamples = 1000; priSamples = 3000;
intervals = [3000 3310 3000 3310 3000 3310 3000]; starts = [0 cumsum(intervals)];
sampleLength = starts(end) + priSamples; backgroundJnrDb = 20; vftJnrDb = 8;
receipt = baseReceipt(executeGeneration, baseSeed, count, fs, pulseSamples, priSamples, intervals, starts, sampleLength, backgroundJnrDb, vftJnrDb);
if ~executeGeneration; writeReceipt(auditRoot, 'dry_run', receipt); return; end
if isfolder(outputRoot) && ~isempty(dir(fullfile(outputRoot, '*')))
    error('RouteA:NoOverwrite', 'Formal H1a source already exists.');
end
if ~isfolder(outputRoot); mkdir(outputRoot); end
absent = makeArm(sampleLength, count); present = makeArm(sampleLength, count);
for index = 1:count
    seed = baseSeed + index - 1;
    [absentItem, presentItem] = makePair(seed, fs, pulseSamples, starts, priSamples, backgroundJnrDb, vftJnrDb);
    absent = store(absent, index, absentItem, seed, false);
    present = store(present, index, presentItem, seed, true);
end
absentPath = fullfile(outputRoot, 'vft_absent_mft_isrj.mat');
presentPath = fullfile(outputRoot, 'vft_present_weak_8db_mft_isrj.mat');
writeArm(absentPath, absent); writeArm(presentPath, present);
receipt.status = 'completed'; receipt.absent_path = absentPath; receipt.present_path = presentPath;
writeReceipt(auditRoot, 'formal', receipt);
end

function receipt = baseReceipt(executed, baseSeed, count, fs, pulseSamples, priSamples, intervals, starts, sampleLength, backgroundJnrDb, vftJnrDb)
receipt = struct('status', 'planned', 'executed', logical(executed), ...
    'experiment', 'RouteA_H1a_VFTStaggeredPRI_MFTISRJCompositeAttribution', ...
    'evidence_role', 'fresh_paired_project_local_staggered_pri_composite_physical_identifiability', ...
    'source_scope', 'project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716', ...
    'base_sample_seed', baseSeed, 'sample_seed_range', [baseSeed baseSeed+count-1], 'pair_count', count, ...
    'fs_Hz', fs, 'pulse_samples', pulseSamples, 'pri_samples', priSamples, 'num_pulses', 8, ...
    'inter_pulse_intervals_samples', intervals, 'pulse_start_samples', starts, 'record_samples', sampleLength, ...
    'background_codes', {{'MFT','ISRJ'}}, 'background_component_effective_JNR_dB', backgroundJnrDb, ...
    'vft_present_effective_JNR_dB', vftJnrDb, 'vft_absent_effective_JNR_dB', NaN, ...
    'vft_doppler_Hz', [1000 7000], 'arrival_delay_samples', [100 500], ...
    'receiver_noise_substream_formula', '(sample_seed + 2*2246822519) mod 2^32', ...
    'component_substreams', struct('mft', 747796405, 'isrj', 2891336453, 'vft', 1181783497), ...
    'test_split_generated', false, 'test_files_opened', false, 'locked_test_used', false, 'test_v2_used', false, 'test_v3_used', false);
if executed; receipt.status = 'completed'; end
end

function arm = makeArm(length, count)
arm = struct('iq', complex(zeros(length, count)), 'pair_seed', zeros(1,count), 'receiver_noise_seed', zeros(1,count), ...
    'vft_present', zeros(1,count), 'mft_component_seed', zeros(1,count), 'isrj_component_seed', zeros(1,count), ...
    'vft_component_seed', nan(1,count), 'mft_num_targets', zeros(1,count), 'mft_delays_samples', nan(8,count), ...
    'mft_amplitudes', nan(8,count), 'isrj_width_samples', zeros(1,count), 'isrj_replicas', zeros(1,count), ...
    'mft_effective_JNR_dB', zeros(1,count), 'isrj_effective_JNR_dB', zeros(1,count), ...
    'vft_effective_JNR_dB', nan(1,count), 'vft_doppler_Hz', nan(1,count), 'vft_arrival_delay_samples', nan(1,count), ...
    'pulse_start_samples', zeros(8,count));
end

function [absent, present] = makePair(seed, fs, pulseSamples, starts, priSamples, backgroundJnrDb, vftJnrDb)
recordLength = starts(end) + priSamples;
noiseSeed = substreamSeed(seed, 2*2246822519); mftSeed = substreamSeed(seed, 747796405);
isrjSeed = substreamSeed(seed, 2891336453); vftSeed = substreamSeed(seed, 1181783497);
noiseStream = RandStream('mt19937ar','Seed',noiseSeed);
noise = sqrt(1/2) * (randn(noiseStream,1,recordLength) + 1i*randn(noiseStream,1,recordLength));
reference = pulseTrain(fs, pulseSamples, starts, recordLength, 0, 0);
[mftRaw, mft] = makeMft(reference, mftSeed); [isrjRaw, isrj] = makeIsrj(reference, isrjSeed);
mft = calibrate(mftRaw, backgroundJnrDb, mft); isrj = calibrate(isrjRaw, backgroundJnrDb, isrj);
absentIq = mft.signal + isrj.signal + noise;
vftStream = RandStream('mt19937ar','Seed',vftSeed); fd = 1000 + 6000*rand(vftStream); arrival = 100 + round(400*rand(vftStream));
vftRaw = pulseTrain(fs, pulseSamples, starts, recordLength, fd, arrival); vft = calibrate(vftRaw, vftJnrDb, struct());
common = struct('receiver_noise_seed',noiseSeed,'mft_component_seed',mftSeed,'isrj_component_seed',isrjSeed, ...
    'mft_num_targets',mft.numTargets,'mft_delays_samples',mft.delays,'mft_amplitudes',mft.amplitudes, ...
    'isrj_width_samples',isrj.width,'isrj_replicas',isrj.replicas,'mft_effective_JNR_dB',mft.effectiveJnr, ...
    'isrj_effective_JNR_dB',isrj.effectiveJnr,'pulse_start_samples',starts(:));
absent = common; absent.iq = absentIq; absent.vft_component_seed = NaN; absent.vft_effective_JNR_dB = NaN; absent.vft_doppler_Hz = NaN; absent.vft_arrival_delay_samples = NaN;
present = common; present.iq = absentIq + vft.signal; present.vft_component_seed = vftSeed; present.vft_effective_JNR_dB = vft.effectiveJnr; present.vft_doppler_Hz = fd; present.vft_arrival_delay_samples = arrival;
end

function output = pulseTrain(fs, pulseSamples, starts, recordLength, fd, arrival)
t = (0:pulseSamples-1)/fs; lfm = exp(1i*pi*(5e6/100e-6)*t.^2); output = complex(zeros(1,recordLength));
for pulse = 1:numel(starts)
    indices = starts(pulse)+arrival+(1:pulseSamples); absoluteTime = (starts(pulse)+arrival+(0:pulseSamples-1))/fs;
    output(indices) = lfm .* exp(1i*2*pi*fd*absoluteTime);
end
end

function [raw, metadata] = makeMft(reference, seed)
stream = RandStream('mt19937ar','Seed',seed); count = randi(stream,[3 8]); choices = randperm(stream,201,count)+299;
amplitudes = 0.5 + rand(stream,1,count); raw = complex(zeros(size(reference)));
for index = 1:count; raw = raw + amplitudes(index)*zeroDelay(reference,choices(index)); end
metadata = struct('numTargets',count,'delays',[choices nan(1,8-count)],'amplitudes',[amplitudes nan(1,8-count)]);
end

function [raw, metadata] = makeIsrj(reference, seed)
stream = RandStream('mt19937ar','Seed',seed); width = randi(stream,[50 200]); replicas = randi(stream,[1 5]);
positions = 0:numel(reference)-1; window = double(mod(positions,(replicas+1)*width) < width); raw = complex(zeros(size(reference)));
for index = 0:replicas; raw = raw + zeroDelay(reference.*window,index*width); end
arrival = 100 + round(400*rand(stream)); raw = zeroDelay(raw,arrival); metadata = struct('width',width,'replicas',replicas);
end

function result = zeroDelay(value, delay)
result = complex(zeros(size(value))); if delay < numel(value); result(delay+1:end) = value(1:end-delay); end
end

function metadata = calibrate(raw, targetDb, metadata)
scale = sqrt(10^(targetDb/10)/max(mean(abs(raw).^2),eps)); metadata.signal = scale*raw;
metadata.effectiveJnr = 10*log10(mean(abs(metadata.signal).^2));
end

function seed = substreamSeed(sampleSeed, salt)
seed = double(mod(uint64(sampleSeed)+uint64(salt),uint64(4294967296)));
end

function arm = store(arm,index,item,seed,present)
arm.iq(:,index) = item.iq(:); arm.pair_seed(index) = seed; arm.receiver_noise_seed(index) = item.receiver_noise_seed; arm.vft_present(index) = double(present);
names = {'mft_component_seed','isrj_component_seed','vft_component_seed','mft_num_targets','isrj_width_samples','isrj_replicas','mft_effective_JNR_dB','isrj_effective_JNR_dB','vft_effective_JNR_dB','vft_doppler_Hz','vft_arrival_delay_samples'};
for k=1:numel(names); arm.(names{k})(index) = item.(names{k}); end
arm.mft_delays_samples(:,index) = item.mft_delays_samples(:); arm.mft_amplitudes(:,index) = item.mft_amplitudes(:); arm.pulse_start_samples(:,index) = item.pulse_start_samples(:);
end

function writeArm(path,arm)
h5create(path,'/iq_real',size(real(arm.iq)),'Datatype','double'); h5write(path,'/iq_real',real(arm.iq));
h5create(path,'/iq_imag',size(imag(arm.iq)),'Datatype','double'); h5write(path,'/iq_imag',imag(arm.iq));
names = {'pair_seed','receiver_noise_seed','vft_present','mft_component_seed','isrj_component_seed','vft_component_seed','mft_num_targets','mft_delays_samples','mft_amplitudes','isrj_width_samples','isrj_replicas','mft_effective_JNR_dB','isrj_effective_JNR_dB','vft_effective_JNR_dB','vft_doppler_Hz','vft_arrival_delay_samples','pulse_start_samples'};
for k=1:numel(names); value=arm.(names{k}); h5create(path,['/' names{k}],size(value),'Datatype','double'); h5write(path,['/' names{k}],double(value)); end
end

function writeReceipt(root,prefix,receipt)
path=fullfile(root,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H1A_RECEIPT=%s\n',path);
end
