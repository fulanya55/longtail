function receipt = materialize_route_a_h5o_ar1_iq(executeMaterialization)
%MATERIALIZE_ROUTE_A_H5O_AR1_IQ Replace only receiver noise on the H5o base.
if nargin < 1; executeMaterialization = false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); dataRoot='C:\LJQ_code\干扰数据';
basePath=fullfile(dataRoot,'复合干扰数据0803_h5o_frozen_unified_ood_base','closedset_6class_train.mat');
derivedPath=fullfile(projectRoot,'outputs','tapc','route_a_h5o_frozen_unified_ood_generation','ar1_iq_payload.mat'); rho=.65;
receipt=struct('status','completed','executed',logical(executeMaterialization), ...
 'experiment','RouteA_H5o_FrozenUnifiedCombinedOOD','evidence_role','project_local_ar1_iq_transform_on_high_replica_base', ...
 'base_path',basePath,'derived_iq_payload',derivedPath,'rho',rho,'noise_power',1, ...
 'transform','base_iq - exact_reference_white(seed) + ar1(seed)','test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h5o_used',false);
if executeMaterialization
 base=load(basePath,'iq_complex','receiver_noise_seed'); [sampleLength,sampleCount]=size(base.iq_complex); iq_complex=complex(zeros(sampleLength,sampleCount)); %#ok<NASGU>
 whiteEnergy=zeros(1,sampleCount); arEnergy=zeros(1,sampleCount);
 for column=1:sampleCount
  seed=double(base.receiver_noise_seed(column)); white=route_a_h0h_white_receiver_noise(seed,sampleLength,1); ar=route_a_h0h_ar1_receiver_noise(seed,sampleLength,1,rho);
  iq_complex(:,column)=base.iq_complex(:,column)-white(:)+ar(:); whiteEnergy(column)=mean(abs(white).^2); arEnergy(column)=mean(abs(ar).^2);
 end
 outRoot=fileparts(derivedPath); if ~isfolder(outRoot); mkdir(outRoot); end; save(derivedPath,'iq_complex','-v7.3');
 receipt.sample_length=sampleLength; receipt.sample_count=sampleCount; receipt.mean_white_power=mean(whiteEnergy); receipt.mean_ar1_power=mean(arEnergy); receipt.ar1_iq_payload_bytes=dir(derivedPath).bytes;
 probe=route_a_h0h_ar1_receiver_noise(1048677743,200000,1,rho); receipt.process_probe=struct('receiver_noise_seed',1048677743,'sample_count',200000,'mean_power',mean(abs(probe).^2),'pseudocovariance_abs',abs(mean(probe.^2)),'lag1_real_correlation',real(mean(conj(probe(1:end-1)).*probe(2:end))),'target_rho',rho,'power_tolerance',.02,'correlation_tolerance',.02);
end
outRoot=fileparts(derivedPath); if ~isfolder(outRoot); mkdir(outRoot); end; prefix='dry_run'; if executeMaterialization; prefix='formal'; end
path=fullfile(outRoot,[prefix '_ar1_materialization_receipt.json']); fid=fopen(path,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
encoded=unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'); fwrite(fid,encoded,'uint8'); fclose(fid); fprintf('ROUTE_A_H5O_AR1_RECEIPT=%s\n',path);
end
