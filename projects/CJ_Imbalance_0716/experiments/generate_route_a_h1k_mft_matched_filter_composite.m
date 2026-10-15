function receipt = generate_route_a_h1k_mft_matched_filter_composite(executeGeneration)
%GENERATE_ROUTE_A_H1K_MFT_MATCHED_FILTER_COMPOSITE Frozen H1k train-only source.
if nargin < 1; executeGeneration = false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hits=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(hits); error('RouteA:GeneratorNotFound','Cannot locate closed-set generator.'); end
generatorRoot=hits(1).folder; dataRoot=fileparts(fileparts(generatorRoot));
outputRoot=fullfile(dataRoot,'复合干扰数据0802_h1k_mft_support_composite');
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h1k_mft_support_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.seed=830000000; cfg.output_dir=outputRoot;
cfg.output.save_legacy=false; cfg.dataset.samples_per_pattern=80; cfg.dataset.num_samples=5040;
cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
result=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
receipt=struct('status','completed','executed',logical(executeGeneration),'experiment','RouteA_H1k_MFTMatchedFilterComposite', ...
 'evidence_role','fresh_composite_mft_fixed_statistic_validation','base_seed',cfg.seed,'output_dir',outputRoot, ...
 'split_selection','train_only','train_path',result.train_path,'train_status',result.splits.train.status, ...
 'sample_count',5040,'samples_per_pattern',80,'expected_receiver_noise_seed_range',[1028677743,1028682782], ...
 'generator_change','none; inherited standard 0716 balanced six-label protocol','test_split_generated',false, ...
 'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h1k_used',false);
if executeGeneration; receipt.train_summary=result.splits.train.summary; end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
path=fullfile(auditRoot,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8');
if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H1K_RECEIPT=%s\n',path);
end
