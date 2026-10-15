function receipt = generate_route_a_h0i_vft_dechirped_doppler(executeGeneration)
if nargin < 1; executeGeneration=false; end
projectRoot=fileparts(fileparts(mfilename('fullpath'))); codeRoot=fileparts(projectRoot);
hit=dir(fullfile(codeRoot,'**','create_closedset_6class_config.m'));
if isempty(hit); error('RouteA:GeneratorNotFound','Cannot locate read-only generator.'); end
generatorRoot=hit(1).folder; dataRoot=fileparts(fileparts(generatorRoot));
auditRoot=fullfile(projectRoot,'outputs','tapc','route_a_h0i_vft_dechirped_doppler_generation');
addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
arms=struct('name',{'low','high'},'doppler',{[1e3,2.5e3],[5.5e3,7e3]}); results=cell(1,numel(arms));
for index=1:numel(arms)
    cfg=create_closedset_6class_config(); cfg.seed=790000000;
    cfg.output_dir=fullfile(dataRoot,['澶嶅悎骞叉壈鏁版嵁0802_h0i_vft_doppler_' arms(index).name]);
    cfg.output.save_legacy=false; cfg.types.enabled={'VFT'}; cfg.types.known={'VFT'}; cfg.types.unknown={};
    cfg.dataset.min_components=1; cfg.dataset.max_components=1; cfg.dataset.combination_mode='all';
    cfg.dataset.samples_per_pattern=640; cfg.dataset.num_samples=640;
    cfg.params.VFT.doppler_Hz=arms(index).doppler;
    cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
    results{index}=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
end
receipt=struct('status','completed','executed',logical(executeGeneration),'experiment','RouteA_H0i_VFTDechirpedDopplerIdentifiability','evidence_role','fresh_paired_vft_doppler_physical_screen','base_seed',790000000,'sample_count_per_arm',640,'low_doppler_Hz',[1e3,2.5e3],'high_doppler_Hz',[5.5e3,7e3],'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_h0i_used',false);
if executeGeneration; receipt.low_path=results{1}.train_path; receipt.high_path=results{2}.train_path; end
if ~isfolder(auditRoot); mkdir(auditRoot); end
prefix='dry_run'; if executeGeneration; prefix='formal'; end
path=fullfile(auditRoot,[prefix '_generation_receipt.json']); fid=fopen(path,'w','n','UTF-8'); cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char'); fprintf('ROUTE_A_H0I_VFT_DOPPLER_RECEIPT=%s\n',path);
end
