function receipt = generate_route_a_h0_isrj_delay_block_self_similarity(executeGeneration)
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot));
outputRoot=fullfile(dataRoot,'复合干扰数据0802_isrj_delay_block_h0');
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0_isrj_delay_block_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config(); cfg.seed=700000000; cfg.output_dir=outputRoot; cfg.output.save_legacy=false;
cfg.dataset.samples_per_pattern=80; cfg.dataset.num_samples=5040;
cfg.params.ISRJ.sample_width_s=[10e-6,30e-6]; cfg.params.ISRJ.num_replicas=[3,8];
cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
result=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
receipt=struct('status','completed','executed',logical(executeGeneration),'experiment','RouteA_H0_ISRJDelayBlockSelfSimilarity','evidence_role','fresh_train_only_isrj_replica_identifiability','base_seed',700000000,'sample_count',5040,'samples_per_pattern',80,'isrj_width_s',[10e-6,30e-6],'isrj_replicas',[3,8],'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h0_used',false);
if executeGeneration; receipt.train_path=result.train_path; end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
fid=fopen(fullfile(auditRoot,[prefix '_generation_receipt.json']),'w','n','UTF-8'); cleaner=onCleanup(@()fclose(fid)); fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_H0_DELAY_BLOCK_RECEIPT=%s\n',fullfile(auditRoot,[prefix '_generation_receipt.json']));
end
