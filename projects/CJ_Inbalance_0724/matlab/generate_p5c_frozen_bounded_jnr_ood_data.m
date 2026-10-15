function result = generate_p5c_frozen_bounded_jnr_ood_data(executeGeneration)
%GENERATE_P5C_FROZEN_BOUNDED_JNR_OOD_DATA Fresh 30-50 dB paired OOD views.
    if nargin < 1, executeGeneration = false; end
    generatorRoot = 'C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
    projectRoot = 'C:\LJQ_code\CJ_Inbalance_0724';
    addpath(generatorRoot, fullfile(generatorRoot, 'core'), fullfile(generatorRoot, 'generators'));
    levels = 30:5:50; outputRoot = fullfile(projectRoot, 'generated_data', 'p5c_frozen_bounded_jnr_ood_seed880000000'); configs=cell(numel(levels),1);
    for index=1:numel(levels)
        cfg=create_power_sweep_6class_config(levels(index)); cfg.seed=880000000; cfg.dataset.samples_per_pattern=60; cfg.dataset.num_samples=63*cfg.dataset.samples_per_pattern;
        cfg.experiment.name='p5c_frozen_bounded_jnr_ood_seed880000000'; cfg.output_dir=fullfile(outputRoot,sprintf('level_%02d__jnr_%ddB',index,levels(index))); configs{index}=cfg;
    end
    result=run_power_sweep_levels(configs,logical(executeGeneration),'train','resume',logical(executeGeneration));
    if executeGeneration, save(fullfile(outputRoot,'generation_receipt.mat'),'result','-v7.3'); end
end
