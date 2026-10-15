function receipt = generate_route_a_h0l_mft_omp_residual(executeGeneration)
% Fresh paired MFT source for a fixed three-atom OMP diagnostic.
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit); error('RouteA:GeneratorNotFound','Missing generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot)); outputRoot=fullfile(dataRoot,'h0l_mft_omp_residual_pairs_20260803'); auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0l_mft_omp_residual_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.types.enabled={'MFT'};cfg.types.known={'MFT'};cfg.types.unknown={};cfg.dataset.min_components=1;cfg.dataset.max_components=1;cfg.dataset.combination_mode='all';cfg=normalize_and_validate_config(cfg,'balanced');reference=build_reference_signal(cfg);
count=640; lowCount=3; highCount=8; baseSeed=850000000; offset=208677743;
receipt=struct('status','completed','executed',logical(executeGeneration),'experiment','RouteA_H0l_MFTOMPResidualIdentifiability','evidence_role','fresh_paired_mft_sparse_attribution_screen','sample_count_per_arm',count,'mft_num_targets',[lowCount highCount],'base_sample_seed',baseSeed,'expected_receiver_noise_seed_range',[baseSeed+offset+1 baseSeed+offset+count],'fixed_dictionary_delay_samples',[400 999],'fixed_omp_atoms',3,'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h0l_used',false);
if executeGeneration
 if isfolder(outputRoot); error('RouteA:OutputExists','No overwrite permitted.'); end; mkdir(outputRoot); low=makearm(cfg.derived.sample_length,count); high=makearm(cfg.derived.sample_length,count);
 for n=1:count
  seed=baseSeed+n; noiseSeed=seed+offset; stream=RandStream('mt19937ar','Seed',seed); jnr=10+20*rand(stream); arrival=round((10e-6+40e-6*rand(stream))*cfg.radar.fs_Hz);
  ds=RandStream('mt19937ar','Seed',seed+1000000); delays=round((30e-6+20e-6*rand(ds,1,8))*cfg.radar.fs_Hz); if numel(unique(delays))~=8; delays=randperm(ds,201,8)+299; end
  as=RandStream('mt19937ar','Seed',seed+2000000); amplitudes=.5+rand(as,1,8); ns=RandStream('mt19937ar','Seed',noiseSeed); noise=sqrt(.5)*(randn(ns,1,cfg.derived.sample_length)+1i*randn(ns,1,cfg.derived.sample_length));
  low.iq(:,n)=compose(reference.train,arrival,delays,amplitudes,noise,jnr,lowCount); high.iq(:,n)=compose(reference.train,arrival,delays,amplitudes,noise,jnr,highCount);
  low=store(low,n,seed,noiseSeed,arrival,jnr,delays,amplitudes,lowCount); high=store(high,n,seed,noiseSeed,arrival,jnr,delays,amplitudes,highCount);
 end
 writearm(fullfile(outputRoot,'mft_count_3.mat'),low);writearm(fullfile(outputRoot,'mft_count_8.mat'),high);receipt.low_path=fullfile(outputRoot,'mft_count_3.mat');receipt.high_path=fullfile(outputRoot,'mft_count_8.mat');
end
if ~isfolder(auditRoot);mkdir(auditRoot);end;prefix='dry_run';if executeGeneration;prefix='formal';end;path=fullfile(auditRoot,[prefix '_generation_receipt.json']);fid=fopen(path,'w','n','UTF-8');cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');fprintf('ROUTE_A_H0L_OMP_RECEIPT=%s\n',path);
end
function a=makearm(length,count);a=struct('iq',complex(zeros(length,count)),'pair_seed',zeros(1,count),'receiver_noise_seed',zeros(1,count),'arrival_delay_samples',zeros(1,count),'JNR_dB',zeros(1,count),'delay_pool_samples',zeros(8,count),'amplitude_pool',zeros(8,count),'mft_num_targets',zeros(1,count));end
function iq=compose(reference,arrival,delays,amp,noise,jnr,count);clean=complex(zeros(size(reference)));for k=1:count;clean=clean+amp(k)*circshift(reference,[0 arrival+delays(k)]);end;scale=sqrt(10^(jnr/10)/max(mean(abs(clean).^2),eps));iq=complex(real(scale*clean+noise),imag(scale*clean+noise));end
function a=store(a,n,seed,noiseSeed,arrival,jnr,delays,amp,count);a.pair_seed(n)=seed;a.receiver_noise_seed(n)=noiseSeed;a.arrival_delay_samples(n)=arrival;a.JNR_dB(n)=jnr;a.delay_pool_samples(:,n)=delays(:);a.amplitude_pool(:,n)=amp(:);a.mft_num_targets(n)=count;end
function writearm(path,a);h5create(path,'/iq_real',size(real(a.iq)),'Datatype','double');h5write(path,'/iq_real',real(a.iq));h5create(path,'/iq_imag',size(imag(a.iq)),'Datatype','double');h5write(path,'/iq_imag',imag(a.iq));names={'pair_seed','receiver_noise_seed','arrival_delay_samples','JNR_dB','delay_pool_samples','amplitude_pool','mft_num_targets'};for k=1:numel(names);v=a.(names{k});h5create(path,['/' names{k}],size(v),'Datatype','double');h5write(path,['/' names{k}],double(v));end;end
