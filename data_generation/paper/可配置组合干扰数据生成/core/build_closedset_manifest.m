function manifest = build_closedset_manifest(cfg, registry, split)
%BUILD_CLOSEDSET_MANIFEST 构建普通或确定性功率扫描闭集清单。
%   MANIFEST = BUILD_CLOSEDSET_MANIFEST(CFG, REGISTRY, SPLIT) 首先调用
%   BUILD_BALANCED_MANIFEST 生成稳定的闭集组合和样本种子。如果 CFG 含有
%   cfg.experiment.power_sweep，则再调用 APPLY_POWER_SWEEP_JNR_PROFILE 写入
%   类型级功率偏置；否则返回与 BUILD_BALANCED_MANIFEST 完全相同的表。
%
%   该薄包装器使普通六类闭集和九挡功率扫描共享同一原子保存执行内核，
%   同时避免在物理生成器或通用平衡 manifest builder 中加入实验特例。

    if has_pure_label_profile(cfg)
        manifest = build_pure_label_manifest(cfg, registry, split);
    else
        manifest = build_balanced_manifest(cfg, registry, split);
    end
    if has_power_sweep_profile(cfg)
        manifest = apply_power_sweep_jnr_profile(manifest, cfg);
    end
end

function present = has_power_sweep_profile(cfg)
%HAS_POWER_SWEEP_PROFILE 判断配置是否显式启用确定性功率扫描剖面。

    present = isstruct(cfg) && isscalar(cfg) && ...
        isfield(cfg, 'experiment') && ...
        isstruct(cfg.experiment) && isscalar(cfg.experiment) && ...
        isfield(cfg.experiment, 'power_sweep');
end

function present = has_pure_label_profile(cfg)
%HAS_PURE_LABEL_PROFILE 判断是否显式启用精确计数纯标签长尾。
    present = isstruct(cfg) && isscalar(cfg) && ...
        isfield(cfg, 'experiment') && isstruct(cfg.experiment) && ...
        isfield(cfg.experiment, 'pure_label_longtail');
end
