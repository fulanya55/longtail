function receipt = generate_route_a_frequency_development_v3(executeGeneration, levels_dB)
% Fresh paired train-only source for Route-A Frequency Representation V3.
if nargin < 1, executeGeneration = false; end
if nargin < 2, levels_dB = [20, 30]; end
if ~isequal(sort(double(levels_dB(:)')), [20, 30])
    error('RouteA:FrequencyV3InvalidLevels', 'levels_dB must be [20 30].');
end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
% Build Chinese path components from Unicode code points because MATLAB on
% this workstation reads UTF-8 .m source as ANSI and would otherwise create
% a mojibake directory (the V2 generator had exactly that failure mode).
dataDir = char([hex2dec('5E72'),hex2dec('6270'),hex2dec('6570'),hex2dec('636E')]);
genDir = char([hex2dec('5E72'),hex2dec('6270'),hex2dec('4FE1'),hex2dec('53F7'),hex2dec('751F'),hex2dec('6210'),hex2dec('7A0B'),hex2dec('5E8F')]);
cfgDir = char([hex2dec('53EF'),hex2dec('914D'),hex2dec('7F6E'),hex2dec('7EC4'),hex2dec('5408'),hex2dec('5E72'),hex2dec('6270'),hex2dec('6570'),hex2dec('636E'),hex2dec('751F'),hex2dec('6210')]);
generatorRoot = fullfile('E:\LJQ_code',dataDir,genDir,cfgDir);
sweepRoot = fullfile('E:\LJQ_code',dataDir,'dual-axis-frequency-representation-v3');
auditRoot = fullfile(projectRoot,'outputs','route_a_frequency_representation_v3','stage1_data_generation');
addpath(generatorRoot, fullfile(generatorRoot,'core'), fullfile(generatorRoot,'generators'));
baseSeed = 3940000000;
standardLevels = 10:5:50;
configs = cell(numel(levels_dB),1);
for k = 1:numel(levels_dB)
    level = double(levels_dB(k));
    cfg = create_power_sweep_6class_config(level);
    cfg.seed = baseSeed;
    levelIndex = find(standardLevels == level,1,'first');
    cfg.output_dir = fullfile(sweepRoot,sprintf('sawfm_level_%02d__jnr_%ddB',levelIndex,level));
    cfg.experiment.name = 'dual-axis-frequency-representation-v3';
    cfg.experiment.power_sweep.fixed_codes = {'FMB','SinFM','MFT','VFT','ISRJ'};
    cfg.experiment.power_sweep.swept_codes = {'SawFM'};
    cfg.experiment.power_sweep.fixed_JNR_dB = 30;
    cfg.experiment.power_sweep.sweep_JNR_dB = level;
    cfg.experiment.power_sweep.standard_levels_dB = standardLevels;
    cfg.experiment.power_sweep.level_index = levelIndex;
    cfg.experiment.power_sweep.paired_seeds = true;
    configs{k} = cfg;
end
result = run_power_sweep_levels(configs,logical(executeGeneration),'train','error',logical(executeGeneration));
receipt = struct('status','completed','executed',logical(executeGeneration), ...
    'experiment','dual-axis-frequency-representation-v3','base_seed',baseSeed, ...
    'levels_dB',double(levels_dB(:)'),'high_visibility_JNR_dB',30, ...
    'low_visibility_JNR_dB',20,'fixed_other_label_JNR_dB',30, ...
    'fixed_codes',{{'FMB','SinFM','MFT','VFT','ISRJ'}},'swept_codes',{{'SawFM'}}, ...
    'split_selection','train_only','existing_policy','error','sweep_root',sweepRoot, ...
    'manifest_path',result.manifest_path,'paired_seed_verified',result.paired_seed_verified, ...
    'expected_samples_per_level',25200,'expected_combination_count',63, ...
    'samples_per_combination',400,'test_views_generated',false, ...
    'locked_test_used',false,'test_v2_used',false,'test_v3_used',false, ...
    'generated_at',char(datetime('now','TimeZone','Asia/Shanghai','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
if ~isfolder(auditRoot), mkdir(auditRoot); end
name = 'dry_run_generation_receipt.json'; if executeGeneration, name = 'formal_generation_receipt.json'; end
fid = fopen(fullfile(auditRoot,name),'w','n','UTF-8');
if fid < 0, error('RouteA:FrequencyV3ReceiptWriteFailed','Cannot write receipt'); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(receipt,PrettyPrint=true),'char');
fprintf('ROUTE_A_FREQUENCY_V3_RECEIPT=%s\n',fullfile(auditRoot,name));
end
