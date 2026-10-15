function receipt = generate_route_a_isrj_parameter_ood_v1(executeGeneration)
%GENERATE_ROUTE_A_ISRJ_PARAMETER_OOD_V1 Train-only ISRJ parameter-OOD evidence.
    if nargin < 1
        executeGeneration = false;
    end
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    outputRoot = 'C:\LJQ_code\干扰数据\复合干扰数据0801_isrj_ood_v1';
    auditRoot = fullfile(projectRoot,'outputs','tapc','route_a_isrj_parameter_ood_v1_generation');
    addpath(generatorRoot,fullfile(generatorRoot,'core'),fullfile(generatorRoot,'generators'));

    cfg = create_closedset_6class_config();
    cfg.seed = 64000000;
    cfg.output_dir = outputRoot;
    cfg.output.save_legacy = false;
    cfg.dataset.samples_per_pattern = 80;
    cfg.dataset.num_samples = 5040;
    cfg.params.ISRJ.sample_width_s = [22e-6,30e-6];
    cfg.params.ISRJ.num_replicas = [6,8];
    cfg = normalize_and_validate_config(cfg,'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    result = run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','resume');

    receipt = struct('status','completed','executed',logical(executeGeneration), ...
        'evidence_role','external_isrj_parameter_ood_v1','base_seed',cfg.seed, ...
        'output_dir',cfg.output_dir,'split_selection','train_only', ...
        'train_path',result.train_path,'train_status',result.splits.train.status, ...
        'sample_count',5040,'samples_per_pattern',80, ...
        'isrj_sample_width_s',cfg.params.ISRJ.sample_width_s, ...
        'isrj_num_replicas',cfg.params.ISRJ.num_replicas, ...
        'reference_isrj_sample_width_s',[5e-6,20e-6], ...
        'reference_isrj_num_replicas',[1,5], ...
        'test_split_generated',false,'locked_test_used',false, ...
        'test_v2_used',false,'test_v3_used',false,'test_v4_used',false, ...
        'test_v5_used',false,'test_v6_used',false);
    if executeGeneration
        receipt.train_summary = result.splits.train.summary;
    end
    if ~isfolder(auditRoot); mkdir(auditRoot); end
    prefix='dry_run'; if executeGeneration; prefix='formal'; end
    receiptPath=fullfile(auditRoot,sprintf('%s_generation_receipt.json',prefix));
    fid=fopen(receiptPath,'w','n','UTF-8');
    if fid<0; error('RouteA:ReceiptWriteFailed','Cannot write receipt.'); end
    cleaner=onCleanup(@() fclose(fid));
    fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
    fprintf('ROUTE_A_ISRJ_OOD_RECEIPT=%s\n',receiptPath);
end
