function receipt = generate_route_a_h0q_vft_staggered_pri(executeGeneration)
% Fresh paired, staggered-PRI VFT H0q source.  It does not create any test split.
if nargin < 1; executeGeneration = false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
dataRoot=fullfile(codeRoot,'干扰数据'); outputRoot=fullfile(dataRoot,'h0q_vft_staggered_pri_20260803');
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0q_vft_staggered_pri_generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end
baseSeed=2600000000; count=640; fs=10e6; pulseSamples=1000; priSamples=3000;
intervals=[3000 3310 3000 3310 3000 3310 3000]; starts=[0 cumsum(intervals)]; sampleLength=starts(end)+priSamples;
lowRange=[1e3 2.5e3]; highRange=[5.5e3 7e3];
receipt=struct('status','planned','executed',logical(executeGeneration), ...
 'experiment','RouteA_H0q_VFTStaggeredPRIPhaseConsistency','evidence_role','fresh_paired_staggered_pri_vft_physical_identifiability', ...
 'base_sample_seed',baseSeed,'sample_seed_range',[baseSeed baseSeed+count-1],'sample_count_per_arm',count, ...
 'fs_Hz',fs,'pulse_samples',pulseSamples,'pri_samples',priSamples,'num_pulses',8, ...
 'inter_pulse_intervals_samples',intervals,'pulse_start_samples',starts,'record_samples',sampleLength, ...
 'low_doppler_Hz',lowRange,'high_doppler_Hz',highRange,'arrival_delay_samples',[100 500],'JNR_dB',[10 30], ...
 'receiver_noise_substream_formula','(sample_seed + 2*2246822519) mod 2^32', ...
 'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false);
if ~executeGeneration; writeReceipt(auditRoot,'dry_run',receipt); return; end
if isfolder(outputRoot) && ~isempty(dir(fullfile(outputRoot,'*'))); error('RouteA:NoOverwrite','H0q output already exists.'); end
if ~isfolder(outputRoot); mkdir(outputRoot); end
low=makeArm(sampleLength,count); high=makeArm(sampleLength,count);
for index=1:count
  seed=baseSeed+index-1;
  [lowItem,highItem]=makePair(seed,fs,pulseSamples,starts,lowRange,highRange);
  low=store(low,index,lowItem,seed); high=store(high,index,highItem,seed);
end
lowPath=fullfile(outputRoot,'vft_low_1p0_2p5kHz.mat'); highPath=fullfile(outputRoot,'vft_high_5p5_7p0kHz.mat');
writeArm(lowPath,low); writeArm(highPath,high); receipt.status='completed'; receipt.low_path=lowPath; receipt.high_path=highPath; writeReceipt(auditRoot,'formal',receipt);
end

function arm=makeArm(length,count)
arm=struct('iq',complex(zeros(length,count)),'pair_seed',zeros(1,count),'receiver_noise_seed',zeros(1,count), ...
 'doppler_Hz',zeros(1,count),'velocity_mps',zeros(1,count),'effective_JNR_dB',zeros(1,count), ...
 'arrival_delay_samples',zeros(1,count),'doppler_quantile',zeros(1,count),'pulse_start_samples',zeros(8,count));
end

function [low,high]=makePair(seed,fs,pulseSamples,starts,lowRange,highRange)
stream=RandStream('mt19937ar','Seed',seed); jnr=10+20*rand(stream); q=rand(stream); arrival=100+round(400*rand(stream));
noiseSeed=mod(uint64(seed)+uint64(2)*uint64(2246822519),uint64(4294967296));
noiseStream=RandStream('mt19937ar','Seed',double(noiseSeed)); length=starts(end)+3000;
noise=sqrt(1/2)*(randn(noiseStream,1,length)+1i*randn(noiseStream,1,length));
lowFd=lowRange(1)+(lowRange(2)-lowRange(1))*q; highFd=highRange(1)+(highRange(2)-highRange(1))*q;
low=makeItem(lowFd,noise,jnr,arrival,q,fs,pulseSamples,starts,double(noiseSeed));
high=makeItem(highFd,noise,jnr,arrival,q,fs,pulseSamples,starts,double(noiseSeed));
end

function item=makeItem(fd,noise,jnr,arrival,q,fs,pulseSamples,starts,noiseSeed)
length=numel(noise); t=(0:pulseSamples-1)/fs; lfm=exp(1i*pi*(5e6/100e-6)*t.^2); clean=complex(zeros(1,length));
for p=1:numel(starts)
  indices=starts(p)+arrival+(1:pulseSamples); absoluteTime=(starts(p)+arrival+(0:pulseSamples-1))/fs;
  clean(indices)=lfm.*exp(1i*2*pi*fd*absoluteTime);
end
scale=sqrt(10^(jnr/10)/max(mean(abs(clean).^2),eps)); iq=scale*clean+noise;
item=struct('iq',complex(real(iq),imag(iq)),'receiver_noise_seed',noiseSeed,'doppler_Hz',fd, ...
 'velocity_mps',fd*3e8/(2*10e9),'effective_JNR_dB',jnr,'arrival_delay_samples',arrival,'doppler_quantile',q,'pulse_start_samples',starts(:));
end

function arm=store(arm,index,item,seed)
arm.iq(:,index)=item.iq(:); arm.pair_seed(index)=seed; arm.receiver_noise_seed(index)=item.receiver_noise_seed;
arm.doppler_Hz(index)=item.doppler_Hz; arm.velocity_mps(index)=item.velocity_mps; arm.effective_JNR_dB(index)=item.effective_JNR_dB;
arm.arrival_delay_samples(index)=item.arrival_delay_samples; arm.doppler_quantile(index)=item.doppler_quantile; arm.pulse_start_samples(:,index)=item.pulse_start_samples;
end

function writeArm(path,arm)
h5create(path,'/iq_real',size(real(arm.iq)),'Datatype','double'); h5write(path,'/iq_real',real(arm.iq));
h5create(path,'/iq_imag',size(imag(arm.iq)),'Datatype','double'); h5write(path,'/iq_imag',imag(arm.iq));
names={'pair_seed','receiver_noise_seed','doppler_Hz','velocity_mps','effective_JNR_dB','arrival_delay_samples','doppler_quantile','pulse_start_samples'};
for k=1:numel(names); value=arm.(names{k}); h5create(path,['/' names{k}],size(value),'Datatype','double'); h5write(path,['/' names{k}],double(value)); end
end

function writeReceipt(root,prefix,receipt)
path=fullfile(root,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H0Q_RECEIPT=%s\n',path);
end
