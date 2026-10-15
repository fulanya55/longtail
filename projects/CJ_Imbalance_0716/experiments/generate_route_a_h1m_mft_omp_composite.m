function receipt = generate_route_a_h1m_mft_omp_composite(executeGeneration)
% Fresh standard six-label, train-only source for the fixed H1m OMP screen.
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m')); if isempty(hit);error('RouteA:GeneratorNotFound','Missing generator.');end
generatorRoot=hit(1).folder;dataRoot=fileparts(fileparts(generatorRoot));outputRoot=fullfile(dataRoot,'h1m_mft_omp_composite_20260803');auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h1m_mft_omp_composite_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
cfg=create_closedset_6class_config();cfg.seed=870000000;cfg.output_dir=outputRoot;cfg.output.save_legacy=false;cfg.dataset.samples_per_pattern=80;cfg.dataset.num_samples=5040;
cfg=normalize_and_validate_config(cfg,'balanced');registry=build_jamming_registry(cfg);schema=build_parameter_schema(registry);result=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
receipt=struct('status','completed','executed',logical(executeGeneration),'experiment','RouteA_H1m_MFTOMPComposite','evidence_role','fresh_composite_mft_sparse_attribution_validation','base_seed',870000000,'sample_count',5040,'samples_per_pattern',80,'expected_receiver_noise_seed_range',[1068677743,1068682782],'fixed_dictionary_delay_samples',[400,999],'fixed_omp_atoms',3,'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h1m_used',false);
if executeGeneration;receipt.train_path=result.train_path;end
if ~isfolder(auditRoot);mkdir(auditRoot);end;prefix='dry_run';if executeGeneration;prefix='formal';end;path=fullfile(auditRoot,[prefix '_generation_receipt.json']);fid=fopen(path,'w','n','UTF-8');cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');fprintf('ROUTE_A_H1M_OMP_RECEIPT=%s\n',path);
end
