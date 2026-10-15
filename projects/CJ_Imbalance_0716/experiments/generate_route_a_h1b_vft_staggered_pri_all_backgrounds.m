function receipt = generate_route_a_h1b_vft_staggered_pri_all_backgrounds(executeGeneration)
% H1b: paired 5-background extension. Formulas follow read-only 0716 code,
% while nonuniform pulse timing makes this a project-local physics extension.
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
dataRoot=fullfile(codeRoot,'干扰数据'); outputRoot=fullfile(dataRoot,'h1b_vft_staggered_pri_all_backgrounds_20260803');
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h1b_vft_staggered_pri_generation'); if ~isfolder(auditRoot); mkdir(auditRoot); end
baseSeed=2800000000; count=640; fs=10e6; pulseSamples=1000; priSamples=3000; intervals=[3000 3310 3000 3310 3000 3310 3000]; starts=[0 cumsum(intervals)]; recordLength=starts(end)+priSamples;
receipt=baseReceipt(executeGeneration,baseSeed,count,fs,pulseSamples,priSamples,intervals,starts,recordLength);
if ~executeGeneration; writeReceipt(auditRoot,'dry_run',receipt); return; end
if isfolder(outputRoot) && ~isempty(dir(fullfile(outputRoot,'*'))); error('RouteA:NoOverwrite','Formal H1b source already exists.'); end
if ~isfolder(outputRoot); mkdir(outputRoot); end
absent=makeArm(recordLength,count); present=makeArm(recordLength,count);
for index=1:count
 seed=baseSeed+index-1; [a,b]=makePair(seed,fs,pulseSamples,starts,priSamples); absent=store(absent,index,a,seed,false); present=store(present,index,b,seed,true);
end
absentPath=fullfile(outputRoot,'vft_absent_all_five_backgrounds.mat'); presentPath=fullfile(outputRoot,'vft_present_weak_8db_all_five_backgrounds.mat');
writeArm(absentPath,absent); writeArm(presentPath,present); receipt.status='completed'; receipt.absent_path=absentPath; receipt.present_path=presentPath; writeReceipt(auditRoot,'formal',receipt);
end

function receipt=baseReceipt(executed,baseSeed,count,fs,pulseSamples,priSamples,intervals,starts,recordLength)
receipt=struct('status','planned','executed',logical(executed),'experiment','RouteA_H1b_VFTStaggeredPRI_AllFiveBackgroundCompositeAttribution', ...
 'evidence_role','fresh_paired_project_local_staggered_pri_all_background_composite_physical_identifiability', ...
 'source_scope','project_local_staggered_pri_physics_extension_not_official_uniform_pri_0716', ...
 'base_sample_seed',baseSeed,'sample_seed_range',[baseSeed baseSeed+count-1],'pair_count',count,'fs_Hz',fs,'pulse_samples',pulseSamples,'pri_samples',priSamples,'num_pulses',8,'inter_pulse_intervals_samples',intervals,'pulse_start_samples',starts,'record_samples',recordLength, ...
 'background_codes',{{'FMB','SawFM','SinFM','MFT','ISRJ'}},'background_component_effective_JNR_dB',20,'vft_present_effective_JNR_dB',8,'vft_absent_effective_JNR_dB',NaN,'vft_doppler_Hz',[1000 7000],'arrival_delay_samples',[100 500], ...
 'receiver_noise_substream_formula','(sample_seed + 2*2246822519) mod 2^32','component_substreams',struct('fmb',747796405,'sawfm',2891336453,'sinfm',1181783497,'mft',3160543729,'isrj',1013904223,'vft',1664525), ...
 'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false);
if executed; receipt.status='completed'; end
end

function arm=makeArm(length,count)
arm=struct('iq',complex(zeros(length,count)),'pair_seed',zeros(1,count),'receiver_noise_seed',zeros(1,count),'vft_present',zeros(1,count), ...
 'background_component_seeds',zeros(5,count),'background_effective_JNR_dB',zeros(5,count),'mft_num_targets',zeros(1,count),'mft_delays_samples',nan(8,count),'mft_amplitudes',nan(8,count),'isrj_width_samples',zeros(1,count),'isrj_replicas',zeros(1,count), ...
 'vft_component_seed',nan(1,count),'vft_effective_JNR_dB',nan(1,count),'vft_doppler_Hz',nan(1,count),'vft_arrival_delay_samples',nan(1,count),'pulse_start_samples',zeros(8,count));
end

function [absent,present]=makePair(seed,fs,pulseSamples,starts,priSamples)
length=starts(end)+priSamples; salts=[2*2246822519 747796405 2891336453 1181783497 3160543729 1013904223 1664525]; ids=arrayfun(@(salt) substreamSeed(seed,salt),salts);
noiseStream=RandStream('mt19937ar','Seed',ids(1)); noise=sqrt(1/2)*(randn(noiseStream,1,length)+1i*randn(noiseStream,1,length));
fmb=calibrate(makeFmb(length,fs,ids(2)),20,struct()); saw=calibrate(makeSaw(length,fs,ids(3)),20,struct()); sinfm=calibrate(makeSin(length,fs,ids(4)),20,struct());
reference=pulseTrain(fs,pulseSamples,starts,length,0,0); [mftRaw,mft]=makeMft(reference,ids(5)); [isrjRaw,isrj]=makeIsrj(reference,ids(6)); mft=calibrate(mftRaw,20,mft); isrj=calibrate(isrjRaw,20,isrj);
background=fmb.signal+saw.signal+sinfm.signal+mft.signal+isrj.signal+noise;
vstream=RandStream('mt19937ar','Seed',ids(7)); fd=1000+6000*rand(vstream); arrival=100+round(400*rand(vstream)); vft=calibrate(pulseTrain(fs,pulseSamples,starts,length,fd,arrival),8,struct());
common=struct('receiver_noise_seed',ids(1),'background_component_seeds',ids(2:6).','background_effective_JNR_dB',[fmb.effectiveJnr;saw.effectiveJnr;sinfm.effectiveJnr;mft.effectiveJnr;isrj.effectiveJnr], ...
 'mft_num_targets',mft.numTargets,'mft_delays_samples',mft.delays,'mft_amplitudes',mft.amplitudes,'isrj_width_samples',isrj.width,'isrj_replicas',isrj.replicas,'pulse_start_samples',starts(:));
absent=common; absent.iq=background; absent.vft_component_seed=NaN; absent.vft_effective_JNR_dB=NaN; absent.vft_doppler_Hz=NaN; absent.vft_arrival_delay_samples=NaN;
present=common; present.iq=background+vft.signal; present.vft_component_seed=ids(7); present.vft_effective_JNR_dB=vft.effectiveJnr; present.vft_doppler_Hz=fd; present.vft_arrival_delay_samples=arrival;
end

function signal=makeFmb(length,fs,seed)
stream=RandStream('mt19937ar','Seed',seed); bandwidth=min((1.5+1.5*rand(stream))*5e6,fs*(1-1e-12)); white=randn(stream,1,length); frequency=(-floor(length/2):ceil(length/2)-1)*(fs/length); spectrum=fftshift(fft(white)); mask=abs(frequency)<=bandwidth/2; colored=real(ifft(ifftshift(spectrum.*mask))); colored=colored-mean(colored); colored=colored/max(abs(colored)); phase=2*pi*cumsum((bandwidth/2)*colored)/fs; signal=circshift(exp(1i*phase),[0,100+round(400*rand(stream))]);
end

function signal=makeSaw(length,fs,seed)
stream=RandStream('mt19937ar','Seed',seed); bandwidth=2e6+3e6*rand(stream); period=20e-6+60e-6*rand(stream); time=(0:length-1)/fs; frequency=-bandwidth/2+bandwidth*mod(time,period)/period; signal=circshift(exp(1i*2*pi*cumsum(frequency)/fs),[0,100+round(400*rand(stream))]);
end

function signal=makeSin(length,fs,seed)
stream=RandStream('mt19937ar','Seed',seed); center=.5e6+1e6*rand(stream); deviation=1e6+2e6*rand(stream); modulation=10e3+10e3*rand(stream); time=(0:length-1)/fs; frequency=center+deviation*cos(2*pi*modulation*time); signal=circshift(exp(1i*2*pi*cumsum(frequency)/fs),[0,100+round(400*rand(stream))]);
end

function output=pulseTrain(fs,pulseSamples,starts,recordLength,fd,arrival)
t=(0:pulseSamples-1)/fs; lfm=exp(1i*pi*(5e6/100e-6)*t.^2); output=complex(zeros(1,recordLength)); for p=1:numel(starts); indices=starts(p)+arrival+(1:pulseSamples); absoluteTime=(starts(p)+arrival+(0:pulseSamples-1))/fs; output(indices)=lfm.*exp(1i*2*pi*fd*absoluteTime); end
end

function [raw,metadata]=makeMft(reference,seed)
stream=RandStream('mt19937ar','Seed',seed); count=randi(stream,[3 8]); choices=randperm(stream,201,count)+299; amplitudes=.5+rand(stream,1,count); raw=complex(zeros(size(reference))); for k=1:count; raw=raw+amplitudes(k)*zeroDelay(reference,choices(k)); end; metadata=struct('numTargets',count,'delays',[choices nan(1,8-count)],'amplitudes',[amplitudes nan(1,8-count)]);
end

function [raw,metadata]=makeIsrj(reference,seed)
stream=RandStream('mt19937ar','Seed',seed); width=randi(stream,[50 200]); replicas=randi(stream,[1 5]); positions=0:numel(reference)-1; window=double(mod(positions,(replicas+1)*width)<width); raw=complex(zeros(size(reference))); for k=0:replicas; raw=raw+zeroDelay(reference.*window,k*width); end; raw=zeroDelay(raw,100+round(400*rand(stream))); metadata=struct('width',width,'replicas',replicas);
end

function result=zeroDelay(value,delay)
result=complex(zeros(size(value))); if delay<numel(value); result(delay+1:end)=value(1:end-delay); end
end

function metadata=calibrate(raw,targetDb,metadata)
metadata.signal=sqrt(10^(targetDb/10)/max(mean(abs(raw).^2),eps))*raw; metadata.effectiveJnr=10*log10(mean(abs(metadata.signal).^2));
end

function seed=substreamSeed(sampleSeed,salt)
seed=double(mod(uint64(sampleSeed)+uint64(salt),uint64(4294967296)));
end

function arm=store(arm,index,item,seed,isPresent)
arm.iq(:,index)=item.iq(:); arm.pair_seed(index)=seed; arm.receiver_noise_seed(index)=item.receiver_noise_seed; arm.vft_present(index)=double(isPresent); arm.background_component_seeds(:,index)=item.background_component_seeds; arm.background_effective_JNR_dB(:,index)=item.background_effective_JNR_dB; arm.mft_num_targets(index)=item.mft_num_targets; arm.mft_delays_samples(:,index)=item.mft_delays_samples(:); arm.mft_amplitudes(:,index)=item.mft_amplitudes(:); arm.isrj_width_samples(index)=item.isrj_width_samples; arm.isrj_replicas(index)=item.isrj_replicas; arm.vft_component_seed(index)=item.vft_component_seed; arm.vft_effective_JNR_dB(index)=item.vft_effective_JNR_dB; arm.vft_doppler_Hz(index)=item.vft_doppler_Hz; arm.vft_arrival_delay_samples(index)=item.vft_arrival_delay_samples; arm.pulse_start_samples(:,index)=item.pulse_start_samples(:);
end

function writeArm(path,arm)
h5create(path,'/iq_real',size(real(arm.iq)),'Datatype','double'); h5write(path,'/iq_real',real(arm.iq)); h5create(path,'/iq_imag',size(imag(arm.iq)),'Datatype','double'); h5write(path,'/iq_imag',imag(arm.iq));
names={'pair_seed','receiver_noise_seed','vft_present','background_component_seeds','background_effective_JNR_dB','mft_num_targets','mft_delays_samples','mft_amplitudes','isrj_width_samples','isrj_replicas','vft_component_seed','vft_effective_JNR_dB','vft_doppler_Hz','vft_arrival_delay_samples','pulse_start_samples'}; for k=1:numel(names); value=arm.(names{k}); h5create(path,['/' names{k}],size(value),'Datatype','double'); h5write(path,['/' names{k}],double(value)); end
end

function writeReceipt(root,prefix,receipt)
path=fullfile(root,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end; cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H1B_RECEIPT=%s\n',path);
end
