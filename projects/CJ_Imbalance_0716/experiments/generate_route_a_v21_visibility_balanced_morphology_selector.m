function receipt = generate_route_a_v21_visibility_balanced_morphology_selector(executeGeneration)
% Fresh train-only source for visibility-balanced frozen morphology selector.
if nargin < 1; executeGeneration = false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot); hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(hit);error('RouteA:GeneratorNotFound','Cannot locate generator.');end
generatorRoot=hit(1).folder;dataRoot=fileparts(fileparts(generatorRoot));outputRoot=fullfile(dataRoot,'复合干扰数据0802_v21_visibility_balanced_selector');auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_v21_visibility_balanced_selector_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config();cfg.seed=600000000;cfg.output_dir=outputRoot;cfg.dataset.samples_per_pattern=80;cfg.dataset.num_samples=5040;cfg.params.ISRJ.sample_width_s=[10e-6,30e-6];cfg.params.ISRJ.num_replicas=[3,4];cfg=normalize_and_validate_config(cfg,'balanced');registry=build_jamming_registry(cfg);schema=build_parameter_schema(registry);result=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
receipt=struct('status','completed','executed',logical(executeGeneration),'evidence_role','fresh_v21_visibility_balanced_morphology_selector','base_seed',600000000,'sample_count',5040,'samples_per_pattern',80,'isrj_width_s',[10e-6,30e-6],'isrj_replicas',[3,4],'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_v21_used',false);
if executeGeneration;receipt.train_path=result.train_path;end
if ~isfolder(auditRoot);mkdir(auditRoot);end;prefix='dry_run';if executeGeneration;prefix='formal';end;path=fullfile(auditRoot,[prefix '_generation_receipt.json']);fid=fopen(path,'w');cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8');fprintf('ROUTE_A_V21_SELECTOR_RECEIPT=%s\n',path);
end
