function result = generate_p5d_sixclass_lsco_data(executeGeneration)
%GENERATE_P5D_SIXCLASS_LSCO_DATA Fresh six-class variable-JNR train-only pool.
    if nargin < 1, executeGeneration = false; end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    cfg=create_closedset_6class_config(); cfg.seed=890000000; cfg.dataset.samples_per_pattern=100; cfg.dataset.num_samples=6300; cfg.params.common.JNR_dB=[10,30];
    cfg.experiment.name='p5d_sixclass_lsco_seed890000000'; cfg.output_dir=fullfile(projectRoot,'generated_data','p5d_sixclass_lsco_seed890000000');
    cfg=normalize_and_validate_config(cfg,'balanced'); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry);
    result=run_closedset_splits(cfg,registry,schema,logical(executeGeneration),'train','resume');
    if executeGeneration, save(fullfile(cfg.output_dir,'generation_receipt.mat'),'result','-v7.3'); end
end
