function receipt = materialize_route_a_h0s_ar1_iq(executeMaterialization)
%MATERIALIZE_ROUTE_A_H0S_AR1_IQ Transform the new H0s train-only base only.
if nargin < 1; executeMaterialization = false; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataRoot = 'C:\LJQ_code\干扰数据';
basePath = fullfile(dataRoot,'复合干扰数据0803_isrj_ood_h0s_r1_ar1_base','closedset_6class_train.mat');
derivedPath = fullfile(projectRoot,'outputs','tapc','route_a_h0s_isrj_combined_generation','ar1_iq_payload.mat');
rho = 0.80;
receipt = struct('status','completed','executed',logical(executeMaterialization), ...
    'experiment','RouteA_H0s_ISRJCombinedParameterNoiseFrozen', ...
    'evidence_role','project_local_ar1_iq_transform_on_high_replica_base', ...
    'base_path',basePath,'derived_iq_payload',derivedPath,'rho',rho, ...
    'noise_power',1,'transform','base_iq - exact_reference_white(seed) + ar1(seed)', ...
    'test_files_opened',false,'locked_test_used',false,'test_v2_used',false, ...
    'test_v3_used',false,'test_h0s_used',false);
if executeMaterialization
    base = load(basePath,'iq_complex','receiver_noise_seed');
    [sampleLength,sampleCount] = size(base.iq_complex);
    iq_complex = complex(zeros(sampleLength,sampleCount)); %#ok<NASGU>
    noisePower = 1;
    whiteEnergy = zeros(1,sampleCount); arEnergy = zeros(1,sampleCount);
    for column=1:sampleCount
        seed = double(base.receiver_noise_seed(column));
        white = route_a_h0h_white_receiver_noise(seed,sampleLength,noisePower);
        ar = route_a_h0h_ar1_receiver_noise(seed,sampleLength,noisePower,rho);
        iq_complex(:,column) = base.iq_complex(:,column) - white(:) + ar(:);
        whiteEnergy(column) = mean(abs(white).^2);
        arEnergy(column) = mean(abs(ar).^2);
    end
    outRoot = fileparts(derivedPath); if ~isfolder(outRoot); mkdir(outRoot); end
    save(derivedPath,'iq_complex','-v7.3');
    receipt.sample_length = sampleLength;
    receipt.sample_count = sampleCount;
    receipt.mean_white_power = mean(whiteEnergy);
    receipt.mean_ar1_power = mean(arEnergy);
    receipt.ar1_iq_payload_bytes = dir(derivedPath).bytes;
    probe = route_a_h0h_ar1_receiver_noise(1008677743,200000,noisePower,rho);
    receipt.process_probe = struct( ...
        'receiver_noise_seed',1008677743,'sample_count',200000, ...
        'mean_power',mean(abs(probe).^2), ...
        'pseudocovariance_abs',abs(mean(probe.^2)), ...
        'lag1_real_correlation',real(mean(conj(probe(1:end-1)).*probe(2:end))), ...
        'target_rho',rho,'power_tolerance',0.02,'correlation_tolerance',0.02);
end
outRoot = fileparts(derivedPath); if ~isfolder(outRoot); mkdir(outRoot); end
prefix='dry_run'; if executeMaterialization; prefix='formal'; end
receiptPath=fullfile(outRoot,[prefix '_ar1_materialization_receipt.json']);
fid=fopen(receiptPath,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H0S_AR1_RECEIPT=%s\n',receiptPath);
end
