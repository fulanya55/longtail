function receipt = generate_route_a_h0j_mft_matched_filter_support(executeGeneration)
% Fresh paired MFT count intervention.  It does not modify reference code.
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(hit); error('RouteA:GeneratorNotFound','Cannot locate read-only generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot));
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0j_mft_matched_filter_support_generation');
outputRoot=fullfile(dataRoot,'h0j_mft_matched_filter_support_pairs_20260802');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.types.enabled={'MFT'}; cfg.types.known={'MFT'}; cfg.types.unknown={};
cfg.dataset.min_components=1; cfg.dataset.max_components=1; cfg.dataset.combination_mode='all';
cfg=normalize_and_validate_config(cfg,'balanced'); reference=build_reference_signal(cfg);
count=640; lowCount=3; highCount=8; baseSeed=810000000; receiverOffset=208677743;
receipt=struct('status','completed','executed',logical(executeGeneration), ...
 'experiment','RouteA_H0j_MFTMatchedFilterSupportIdentifiability', ...
 'evidence_role','fresh_paired_mft_count_physical_screen','sample_count_per_arm',count, ...
 'mft_num_targets',[lowCount highCount],'base_sample_seed',baseSeed, ...
 'expected_receiver_noise_seed_range',[baseSeed+receiverOffset+1 baseSeed+receiverOffset+count], ...
 'fixed_roi_samples',[400 1000],'reference_pulse_samples',1000, ...
 'test_split_generated',false,'test_files_opened',false,'locked_test_used',false, ...
 'test_v2_used',false,'test_v3_used',false,'test_h0j_used',false);
if executeGeneration
 if isfolder(outputRoot); error('RouteA:OutputExists','H0j output already exists; no overwrite permitted.'); end
 mkdir(outputRoot); low=allocate_arm(cfg.derived.sample_length,count); high=allocate_arm(cfg.derived.sample_length,count);
 for index=1:count
  pairSeed=baseSeed+index; noiseSeed=pairSeed+receiverOffset; row=make_pair_row(cfg,reference,pairSeed,noiseSeed);
  low.iq(:,index)=build_arm(cfg,reference,row,lowCount); high.iq(:,index)=build_arm(cfg,reference,row,highCount);
  low=store_row(low,index,row,lowCount); high=store_row(high,index,row,highCount);
 end
 write_arm(fullfile(outputRoot,'mft_count_3.mat'),low); write_arm(fullfile(outputRoot,'mft_count_8.mat'),high);
 receipt.low_path=fullfile(outputRoot,'mft_count_3.mat'); receipt.high_path=fullfile(outputRoot,'mft_count_8.mat');
end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
path=fullfile(auditRoot,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H0J_MFT_SUPPORT_RECEIPT=%s\n',path);
end

function arm=allocate_arm(length,count)
arm=struct('iq',complex(zeros(length,count)),'pair_seed',zeros(1,count),'receiver_noise_seed',zeros(1,count), ...
 'arrival_delay_samples',zeros(1,count),'JNR_dB',zeros(1,count),'delay_pool_samples',zeros(8,count), ...
 'amplitude_pool',zeros(8,count),'mft_num_targets',zeros(1,count));
end

function row=make_pair_row(cfg,reference,pairSeed,noiseSeed)
% Independent deterministic substreams preserve every non-count nuisance.
fs=cfg.radar.fs_Hz; row.pair_seed=pairSeed; row.receiver_noise_seed=noiseSeed;
s=RandStream('mt19937ar','Seed',pairSeed); row.JNR_dB=10+20*rand(s); row.arrival_delay_samples=round((10e-6+40e-6*rand(s))*fs);
s=RandStream('mt19937ar','Seed',pairSeed+1000000); row.delay_pool_samples=round((30e-6+20e-6*rand(s,1,8))*fs);
if numel(unique(row.delay_pool_samples))~=8; row.delay_pool_samples=randperm(s,201,8)+299; end
s=RandStream('mt19937ar','Seed',pairSeed+2000000); row.amplitude_pool=0.5+rand(s,1,8);
s=RandStream('mt19937ar','Seed',noiseSeed); n=cfg.derived.sample_length; row.noise=sqrt(cfg.radar.noise_power/2)*(randn(s,1,n)+1i*randn(s,1,n));
% All copies use the exact project reference and standard JNR power convention.
row.reference=reference.train;
end

function iq=build_arm(cfg,reference,row,targetCount)
clean=complex(zeros(size(reference.train)));
for k=1:targetCount
 delay=row.arrival_delay_samples+row.delay_pool_samples(k); clean=clean+row.amplitude_pool(k)*circshift(reference.train,[0 delay]);
end
power=mean(abs(clean).^2); scale=sqrt(10^(row.JNR_dB/10)/max(power,eps)); iq=complex(real(scale*clean+row.noise),imag(scale*clean+row.noise));
end

function arm=store_row(arm,index,row,targetCount)
arm.pair_seed(index)=row.pair_seed; arm.receiver_noise_seed(index)=row.receiver_noise_seed; arm.arrival_delay_samples(index)=row.arrival_delay_samples;
arm.JNR_dB(index)=row.JNR_dB; arm.delay_pool_samples(:,index)=row.delay_pool_samples(:); arm.amplitude_pool(:,index)=row.amplitude_pool(:); arm.mft_num_targets(index)=targetCount;
end

function write_arm(path,arm)
h5create(path,'/iq_real',size(real(arm.iq)),'Datatype','double'); h5write(path,'/iq_real',real(arm.iq));
h5create(path,'/iq_imag',size(imag(arm.iq)),'Datatype','double'); h5write(path,'/iq_imag',imag(arm.iq));
names={'pair_seed','receiver_noise_seed','arrival_delay_samples','JNR_dB','delay_pool_samples','amplitude_pool','mft_num_targets'};
for k=1:numel(names); value=arm.(names{k}); h5create(path,['/' names{k}],size(value),'Datatype','double'); h5write(path,['/' names{k}],double(value)); end
end
