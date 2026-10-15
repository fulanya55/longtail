function receipt = generate_route_a_h3_frozen_confirmation_dataset(executeGeneration)
% H3 train-only standard source for a frozen candidate; no test split exists.
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot)); outputRoot=fullfile(dataRoot,'复合干扰数据0803_h3_frozen_unified_confirmation_r1');
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h3_frozen_confirmation_generation'); if ~isfolder(auditRoot); mkdir(auditRoot); end
if isfile(fullfile(outputRoot,'closedset_6class_train.mat')); error('RouteA:NoOverwrite','Formal H3 source already exists.'); end
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.seed=134000000; cfg.output_dir=outputRoot; cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
result=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','resume'); count=double(cfg.dataset.num_samples); current=[double(cfg.seed)+1,double(cfg.seed)+count];
receipt=struct('status','completed','executed',logical(executeGeneration),'experiment','RouteA_H3_FrozenUnifiedExternalConfirmation','evidence_role','fresh_standard_train_only_external_confirmation','base_sample_seed',cfg.seed,'sample_seed_range',current,'sample_count',count,'output_dir',cfg.output_dir,'train_path',result.train_path,'train_status',result.splits.train.status,'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false);
if logical(executeGeneration); receipt.train_summary=result.splits.train.summary; end
prefix='dry_run'; if logical(executeGeneration); prefix='formal'; end; path=fullfile(auditRoot,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end; cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H3_RECEIPT=%s\n',path);
end
