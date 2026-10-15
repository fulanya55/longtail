function receipt = generate_route_a_h0p_mft_delay_autocorrelation(executeGeneration)
% Fresh paired single-MFT source for Route-A H0p (no reference is used later).
if nargin < 1; executeGeneration = false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit); error('RouteA:GeneratorNotFound','Missing generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot)); outputRoot=fullfile(dataRoot,'h0p_mft_delay_autocorrelation_pairs_20260803'); auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0p_mft_delay_autocorrelation_generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end
baseSeed=2300000000; count=640; lowCount=3; highCount=8;
receipt=struct('status','planned','executed',logical(executeGeneration),'experiment','RouteA_H0p_MFTDelayAutocorrelationIdentifiability','evidence_role','fresh_single_PRI_no_reference_paired_physical_identifiability','base_sample_seed',baseSeed,'sample_seed_range',[baseSeed baseSeed+count-1],'sample_count_per_arm',count,'mft_num_targets',[lowCount highCount],'common_JNR_dB',[10 30],'mft_delay_samples',[400 999],'num_pulses',1,'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false);
if ~executeGeneration; write_receipt(auditRoot,'dry_run',receipt); return; end
if isfolder(outputRoot) && ~isempty(dir(fullfile(outputRoot,'*'))); error('RouteA:NoOverwrite','H0p source already exists.'); end
if ~isfolder(outputRoot);mkdir(outputRoot);end
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config();cfg.types.enabled={'MFT'};cfg.types.known={'MFT'};cfg.types.unknown={};cfg.dataset.min_components=1;cfg.dataset.max_components=1;cfg.dataset.combination_mode='all';cfg.params.common.JNR_dB=[10 30];cfg=normalize_and_validate_config(cfg,'balanced');reference=build_reference_signal(cfg);
low=make_arm(cfg.derived.sample_length,count); high=make_arm(cfg.derived.sample_length,count);
for index=1:count
 seed=baseSeed+index-1;
 [lowItem,highItem]=make_pair(cfg,reference,seed,lowCount,highCount);
 low=store(low,index,lowItem,seed,lowCount); high=store(high,index,highItem,seed,highCount);
end
lowPath=fullfile(outputRoot,'mft_count_3.mat');highPath=fullfile(outputRoot,'mft_count_8.mat');write_arm(lowPath,low);write_arm(highPath,high);receipt.status='completed';receipt.low_path=lowPath;receipt.high_path=highPath;write_receipt(auditRoot,'formal',receipt);
end
function arm=make_arm(sampleLength,count)
arm=struct('iq',complex(zeros(sampleLength,count)),'pair_seed',zeros(1,count), ...
    'receiver_noise_seed',zeros(1,count),'mft_num_targets',zeros(1,count), ...
    'component_seed',zeros(1,count),'effective_JNR_dB',zeros(1,count));
end
function [low,high]=make_pair(cfg,reference,seed,lowCount,highCount)
% This explicit construction preserves the official MFT waveform, delay
% support and RMS/JNR calibration while making the 3-vs-8 intervention exact.
fs=cfg.radar.fs_Hz; stream=RandStream('mt19937ar','Seed',seed); jnr=10+20*rand(stream); arrival=round((10e-6+40e-6*rand(stream))*fs);
delayStream=RandStream('mt19937ar','Seed',seed+1000000); delays=round((30e-6+20e-6*rand(delayStream,1,highCount))*fs); if numel(unique(delays))~=highCount; delays=randperm(delayStream,201,highCount)+299; end
amplitudeStream=RandStream('mt19937ar','Seed',seed+2000000); amplitudes=.5+rand(amplitudeStream,1,highCount);
noiseSeed=mod(uint64(seed)+uint64(2)*uint64(2246822519),uint64(4294967296)); noiseStream=RandStream('mt19937ar','Seed',double(noiseSeed)); noise=sqrt(cfg.radar.noise_power/2)*(randn(noiseStream,1,cfg.derived.sample_length)+1i*randn(noiseStream,1,cfg.derived.sample_length));
componentSeed=uint64(seed)+uint64(2246822519)+uint64(2654435761)*uint64(6);
low=make_item(reference.train,arrival,delays,amplitudes,noise,jnr,lowCount,double(noiseSeed),componentSeed);
high=make_item(reference.train,arrival,delays,amplitudes,noise,jnr,highCount,double(noiseSeed),componentSeed);
end
function item=make_item(reference,arrival,delays,amplitudes,noise,jnr,count,noiseSeed,componentSeed)
clean=complex(zeros(size(reference))); for k=1:count; clean=clean+amplitudes(k)*circshift(reference,[0 arrival+delays(k)]); end
scale=sqrt(10^(jnr/10)/max(mean(abs(clean).^2),eps)); item=struct('iq_complex',complex(real(scale*clean+noise),imag(scale*clean+noise)),'receiver_noise_seed',noiseSeed,'component_seed',double(mod(componentSeed,uint64(4294967296))),'effective_JNR_dB',jnr);
end
function arm=store(arm,index,item,seed,count)
arm.iq(:,index)=item.iq_complex(:); arm.pair_seed(index)=seed;
arm.receiver_noise_seed(index)=item.receiver_noise_seed; arm.mft_num_targets(index)=count;
arm.component_seed(index)=item.component_seed; arm.effective_JNR_dB(index)=item.effective_JNR_dB;
end
function write_arm(path,arm)
h5create(path,'/iq_real',size(real(arm.iq)),'Datatype','double'); h5write(path,'/iq_real',real(arm.iq));
h5create(path,'/iq_imag',size(imag(arm.iq)),'Datatype','double'); h5write(path,'/iq_imag',imag(arm.iq));
names={'pair_seed','receiver_noise_seed','mft_num_targets','component_seed','effective_JNR_dB'};
for k=1:numel(names)
    v=arm.(names{k}); h5create(path,['/' names{k}],size(v),'Datatype','double'); h5write(path,['/' names{k}],double(v));
end
end
function write_receipt(root,prefix,receipt)
path=fullfile(root,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8');
if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H0P_RECEIPT=%s\n',path);
end
