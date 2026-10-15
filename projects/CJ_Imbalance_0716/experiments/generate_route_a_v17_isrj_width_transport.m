function receipt = generate_route_a_v17_isrj_width_transport(executeGeneration)
%GENERATE_ROUTE_A_V17_ISRJ_WIDTH_TRANSPORT Fresh paired narrow/wide transport data.
    if nargin < 1; executeGeneration = false; end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    codeRoot = fileparts(projectRoot);
    hit = dir(fullfile(codeRoot, '**', 'create_closedset_6class_config.m'));
    if isempty(hit); error('RouteA:GeneratorNotFound','Cannot locate generator.'); end
    generatorRoot = hit(1).folder; dataRoot = fileparts(fileparts(generatorRoot));
    outputRoot = fullfile(dataRoot, '复合干扰数据0802_v17_isrj_width_transport');
    auditRoot = fullfile(projectRoot,'outputs','tapc','route_a_v17_isrj_width_transport_generation');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));
    names={'W10_15_R3_4','W22_30_R3_4'}; widths={[10e-6,15e-6],[22e-6,30e-6]}; result=cell(2,1);
    for i=1:2
        cfg=create_closedset_6class_config(); cfg.seed=530000000; cfg.output_dir=fullfile(outputRoot,names{i});
        cfg.dataset.samples_per_pattern=20; cfg.dataset.num_samples=1260; cfg.params.ISRJ.sample_width_s=widths{i}; cfg.params.ISRJ.num_replicas=[3,4];
        cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
        result{i}=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','error');
    end
    receipt=struct('status','completed','executed',logical(executeGeneration),'evidence_role','fresh_external_predicted_width_transport', ...
        'base_seed',530000000,'regimes',{names},'sample_count_per_regime',1260,'samples_per_pattern',20, ...
        'receiver_noise_seed_range',[728677743,728679002],'widths_s',{widths},'replicas',[3,4], ...
        'test_split_generated',false,'test_files_opened',false,'locked_test_used',false,'test_v2_used',false,'test_v3_used',false,'test_v17_used',false);
    if executeGeneration; receipt.train_paths=cellfun(@(row)row.train_path,result,'UniformOutput',false); end
    if ~isfolder(auditRoot); mkdir(auditRoot); end; prefix='dry_run'; if executeGeneration; prefix='formal'; end
    path=fullfile(auditRoot,[prefix '_generation_receipt.json']); fid=fopen(path,'w'); cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fwrite(fid,unicode2native(jsonencode(receipt,PrettyPrint=true),'UTF-8'),'uint8'); fprintf('ROUTE_A_V17_WIDTH_TRANSPORT_RECEIPT=%s\n',path);
end
