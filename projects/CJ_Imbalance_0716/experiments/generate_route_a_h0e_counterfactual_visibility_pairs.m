function receipt = generate_route_a_h0e_counterfactual_visibility_pairs(executeGeneration)
% Generate paired views whose only changed component is MFT, VFT, or ISRJ.
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot)); outputRoot=fullfile(dataRoot,'复合干扰数据0802_counterfactual_visibility_h0e'); auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0e_counterfactual_visibility_generation');
if ~isfolder(auditRoot); mkdir(auditRoot); end
if ~executeGeneration
    receipt=base_receipt(false,'',0); write_receipt(auditRoot,'dry_run',receipt); return;
end
dataPath=fullfile(outputRoot,'counterfactual_visibility_pairs_train.mat'); if isfile(dataPath); error('RouteA:NoOverwrite','H0e source already exists and must not be overwritten.'); end
if ~isfolder(outputRoot); mkdir(outputRoot); end
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.seed=740000000; cfg.output.save_legacy=false; cfg.params.common.JNR_dB=[20,20]; cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry); reference=build_reference_signal(cfg);
targets={'MFT','VFT','ISRJ'}; nuisance=cell(3,5); nuisance(1,:)={{'FMB','SawFM'},{'FMB','VFT'},{'SawFM','ISRJ'},{'VFT','ISRJ'},{'FMB','SawFM','VFT'}}; nuisance(2,:)={{'FMB','SawFM'},{'FMB','MFT'},{'SawFM','ISRJ'},{'MFT','ISRJ'},{'FMB','SawFM','MFT'}}; nuisance(3,:)={{'FMB','SawFM'},{'FMB','VFT'},{'SawFM','VFT'},{'MFT','VFT'},{'FMB','SawFM','VFT'}};
pairCount=2400; viewCount=2*pairCount; sampleLength=cfg.derived.sample_length; iq_complex=complex(zeros(sampleLength,viewCount)); label_selected=zeros(numel(cfg.types.enabled),viewCount,'uint8'); receiver_noise_seed=zeros(1,viewCount,'uint32'); sample_seed=zeros(1,viewCount,'uint32'); pair_id=zeros(1,viewCount,'uint32'); pair_target_index=zeros(1,viewCount,'uint8'); pair_side=zeros(1,viewCount,'uint8'); weak_target=zeros(1,viewCount,'uint8');
for pairIndex=1:pairCount
    targetPosition=mod(pairIndex-1,3)+1; targetCode=targets{targetPosition}; combo=nuisance{targetPosition,mod(floor((pairIndex-1)/3),5)+1}; seed=cfg.seed+pairIndex-1;
    negativeRow=make_row(pairIndex*2-1,seed,combo,{},registry); positiveRow=make_row(pairIndex*2,seed,combo,{targetCode},registry);
    negative=compose_jamming_sample(cfg,registry,schema,negativeRow,reference); positive=compose_jamming_sample(cfg,registry,schema,positiveRow,reference);
    if negative.receiver_noise_seed~=positive.receiver_noise_seed; error('RouteA:PairNoiseMismatch','paired views must share receiver noise'); end
    columns=[2*pairIndex-1,2*pairIndex]; samples={negative,positive};
    for localIndex=1:2
        column=columns(localIndex); item=samples{localIndex}; iq_complex(:,column)=item.iq_complex(:); label_selected(:,column)=uint8(item.label_selected(:)); receiver_noise_seed(column)=uint32(item.receiver_noise_seed); sample_seed(column)=uint32(item.sample_seed); pair_id(column)=uint32(pairIndex); pair_target_index(column)=uint8(find(strcmp(cfg.types.enabled,targetCode),1)); pair_side(column)=uint8(localIndex-1); weak_target(column)=uint8(localIndex==2);
    end
end
save(dataPath,'iq_complex','label_selected','receiver_noise_seed','sample_seed','pair_id','pair_target_index','pair_side','weak_target','-v7.3');
receipt=base_receipt(true,dataPath,pairCount); write_receipt(auditRoot,'formal',receipt);
end

function row=make_row(sampleId,sampleSeed,nuisance,target,registry)
codes=[nuisance,target]; registryCodes={registry.code}; ids=cellfun(@(code)find(strcmp(registryCodes,code),1),codes); [ids,order]=sort(ids); codes=codes(order); offsets=zeros(1,numel(codes)); weak=false(1,numel(codes)); targetMask=false(1,numel(codes)); if ~isempty(target); targetMask=strcmp(codes,target{1}); offsets(targetMask)=-12; weak(targetMask)=true; end
row=table(double(sampleId),{'train'},double(sampleSeed),{double(ids)},{codes},{strjoin(codes,'+')},double(numel(codes)),{offsets},{weak},1,1,{[false,false,false]},false,0,'VariableNames',{'sample_id','split','sample_seed','active_type_ids','active_type_codes','combination_id','combination_size','component_target_power_dB','weak_component_mask','component_sampling_weight','pattern_sampling_weight','imbalance_flags','unknown_present','unknown_component_count'});
end

function receipt=base_receipt(executed,dataPath,pairCount)
receipt=struct('status','completed','executed',logical(executed),'experiment','RouteA_H0e_CounterfactualVisibilityMargin','evidence_role','fresh_train_only_counterfactual_pair_identifiability','base_seed',740000000,'pair_count',pairCount,'view_count',2*pairCount,'targets',{{'MFT','VFT','ISRJ'}},'common_JNR_dB',[20,20],'target_offset_dB',-12,'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h0e_used',false); if executed; receipt.data_path=dataPath; end
end

function write_receipt(root,prefix,receipt)
fid=fopen(fullfile(root,[prefix '_generation_receipt.json']),'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end; cleaner=onCleanup(@()fclose(fid)); fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H0E_COUNTERFACTUAL_RECEIPT=%s\n',fullfile(root,[prefix '_generation_receipt.json']));
end
