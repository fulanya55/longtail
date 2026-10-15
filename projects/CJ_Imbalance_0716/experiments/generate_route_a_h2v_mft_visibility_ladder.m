function receipt = generate_route_a_h2v_mft_visibility_ladder(executeGeneration)
% Fresh H2v source: a preregistered visibility ladder, not an H1u retry.
if nargin < 1; executeGeneration = false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit); error('RouteA:GeneratorNotFound','Missing official generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot)); outputRoot=fullfile(dataRoot,'复合干扰数据0803_h2v_mft_visibility_ladder_r1');
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h2v_visibility_ladder_generation'); if ~isfolder(auditRoot); mkdir(auditRoot); end
baseSeed=3920000000; perCombination=80; total=63*perCombination; receipt=base_receipt(executeGeneration,baseSeed,total,perCombination);
if ~executeGeneration; write_receipt(auditRoot,'dry_run',receipt); return; end
if isfile(fullfile(outputRoot,'closedset_6class_train.mat')); error('RouteA:NoOverwrite','Formal H2v source already exists.'); end
if ~isfolder(outputRoot); mkdir(outputRoot); end
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.radar.pri_s=1e-3; cfg.radar.num_pulses=1; cfg.output.save_legacy=false; cfg.params.common.JNR_dB=[20,20]; cfg.params.MFT.num_targets=[3,3]; cfg.params.MFT.target_delay_s=[160e-6,700e-6]; cfg=normalize_and_validate_config(cfg,'balanced');
registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry); reference=build_reference_signal(cfg); payload=make_payload(cfg.derived.sample_length,total);
selectedCodes={'FMB','SawFM','SinFM','MFT','VFT','ISRJ'}; rowIndex=0;
for mask=1:63
    active=selectedCodes(logical(bitget(mask,1:6))); background=active(~strcmp(active,'MFT'));
    for repetition=1:perCombination
        rowIndex=rowIndex+1; sampleSeed=baseSeed+rowIndex-1;
        if isempty(background); item=noise_only_item(cfg,sampleSeed,registry); else; item=compose_jamming_sample(cfg,registry,schema,make_row(rowIndex,sampleSeed,background,registry),reference); end
        labels=item.label_selected; weak=false; strength=NaN; delaySamples=nan(1,3); amplitude=nan(1,3); componentSeed=NaN;
        if any(strcmp(active,'MFT'))
            [weak,strength]=visibility_level(repetition); [signal,delaySamples,amplitude,componentSeed]=separated_mft(reference.train,sampleSeed,strength); item.iq_complex=item.iq_complex+signal; labels(4)=true;
        end
        payload=store(payload,rowIndex,item,labels,mask,weak,strength,delaySamples,amplitude,componentSeed,background,registry);
    end
end
path=fullfile(outputRoot,'closedset_6class_train.mat'); write_payload(path,payload); receipt.train_path=path; receipt.sample_length=cfg.derived.sample_length; write_receipt(auditRoot,'formal',receipt);
end

function [weak,strength]=visibility_level(repetition)
if repetition<=14; weak=true; strength=0; elseif repetition<=27; weak=true; strength=4; elseif repetition<=40; weak=true; strength=8; else; weak=false; strength=20; end
end
function payload=make_payload(length,count)
payload=struct('iq',complex(zeros(length,count,'single')),'label_selected',zeros(6,count,'uint8'),'pairing_key',zeros(1,count),'receiver_noise_seed',zeros(1,count),'combination_mask',zeros(1,count),'mft_weak_mask',zeros(1,count),'mft_total_jnr_db',nan(1,count),'mft_delay_samples',nan(3,count),'mft_amplitude',nan(3,count),'mft_component_seed',nan(1,count),'background_component_seeds',nan(5,count),'num_pulses',ones(1,count),'samples_per_pri',10000*ones(1,count));
end
function payload=store(payload,index,item,labels,mask,weak,strength,delays,amplitude,componentSeed,background,registry)
payload.iq(:,index)=single(item.iq_complex(:)); payload.label_selected(:,index)=uint8(labels(:)); payload.pairing_key(index)=item.sample_seed; payload.receiver_noise_seed(index)=item.receiver_noise_seed; payload.combination_mask(index)=mask; payload.mft_weak_mask(index)=double(weak); payload.mft_total_jnr_db(index)=strength; payload.mft_delay_samples(:,index)=delays(:); payload.mft_amplitude(:,index)=amplitude(:); payload.mft_component_seed(index)=componentSeed;
for k=1:numel(background); position=find(strcmp(item.active_type_codes,background{k}),1); payload.background_component_seeds(k,index)=item.component_seeds(position); end
end
function row=make_row(sampleId,sampleSeed,codes,registry)
ids=cellfun(@(code)find(strcmp({registry.code},code),1),codes); [ids,order]=sort(ids); codes=codes(order); offsets=zeros(1,numel(codes)); weak=false(1,numel(codes));
row=table(double(sampleId),{'train'},double(sampleSeed),{double(ids)},{codes},{strjoin(codes,'+')},double(numel(codes)),{offsets},{weak},1,1,{[false,false,false]},false,0,'VariableNames',{'sample_id','split','sample_seed','active_type_ids','active_type_codes','combination_id','combination_size','component_target_power_dB','weak_component_mask','component_sampling_weight','pattern_sampling_weight','imbalance_flags','unknown_present','unknown_component_count'});
end
function item=noise_only_item(cfg,sampleSeed,registry)
modulus=uint64(4294967296); seed=double(mod(uint64(sampleSeed)+2*uint64(2246822519),modulus)); stream=RandStream('mt19937ar','Seed',seed); length=cfg.derived.sample_length;
item=struct('iq_complex',sqrt(cfg.radar.noise_power/2)*(randn(stream,1,length)+1i*randn(stream,1,length)),'label_selected',false(1,6),'sample_seed',sampleSeed,'receiver_noise_seed',seed,'active_type_codes',{{}},'component_seeds',[]);
end
function [signal,delays,amplitude,componentSeed]=separated_mft(reference,sampleSeed,jnrDb)
componentSeed=mod(double(sampleSeed)+170000003,2^32); stream=RandStream('mt19937ar','Seed',componentSeed); delays=[2000,4000,6000]+randi(stream,[-200,200],1,3); amplitude=.5+rand(stream,1,3); signal=complex(zeros(size(reference))); for k=1:3; signal=signal+amplitude(k)*circshift(reference,[0,delays(k)]); end; signal=signal*sqrt(10^(jnrDb/10)/max(mean(abs(signal).^2),eps));
end
function write_payload(path,payload)
h5create(path,'/iq_real',size(real(payload.iq)),'Datatype','single'); h5write(path,'/iq_real',real(payload.iq)); h5create(path,'/iq_imag',size(imag(payload.iq)),'Datatype','single'); h5write(path,'/iq_imag',imag(payload.iq));
names={'label_selected','pairing_key','receiver_noise_seed','combination_mask','mft_weak_mask','mft_total_jnr_db','mft_delay_samples','mft_amplitude','mft_component_seed','background_component_seeds','num_pulses','samples_per_pri'}; for k=1:numel(names); value=payload.(names{k}); h5create(path,['/' names{k}],size(value),'Datatype','double'); h5write(path,['/' names{k}],double(value)); end
end
function receipt=base_receipt(executed,baseSeed,total,perCombination)
receipt=struct('status','planned','executed',logical(executed),'experiment','RouteA_H2v_ConditionalComplementarityAdmission','evidence_role','fresh_external_visibility_ladder_admission','base_sample_seed',baseSeed,'sample_seed_range',[baseSeed,baseSeed+total-1],'sample_count',total,'samples_per_combination',perCombination,'mft_weak_per_mft_combination',40,'mft_strong_per_mft_combination',40,'weak_jnr_counts',struct('jnr_0db',14,'jnr_4db',13,'jnr_8db',13),'mft_strong_jnr_db',20,'pri_us',1000,'num_pulses',1,'mft_minimum_spacing_samples',1600,'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false); if executed; receipt.status='completed'; end
end
function write_receipt(root,prefix,receipt)
path=fullfile(root,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H2V_RECEIPT=%s\n',path);
end
