function gen_dataset_by_marginal_ratio()
clc; clear; close all;
rng('shuffle');

%% ===================== 0) 你的原始信号参数（按你原程序保留） =====================
C = 3e8;
Tp = 100e-6;
Tpp = 300e-6;
fs = 10e6;
Bw = 5e6;
Np = Tp * fs;
nPP = ceil(Tpp * fs);
K = Bw / Tp;

tp = (1 : Np)' / fs;
sigLFM = exp(1i*pi*K*tp.^2);
sig_LOS = [sigLFM ; zeros(ceil(nPP-Np), 1)];

% 干扰全局参数
nPulse = 1;
nRepetition = 1;
flag_addTar = 0;
freq0 = 0;
jam_delay_min = 10e-6;
jam_delay_max = 50e-6;

% 你原来用到的功率参数（请按你的实验设定填写）
J_s_min = 30;  J_s_max = 50;
J_d_min = 20;  J_d_max = 40;

%% ===================== 1) 配置：用单一干扰比例约束复合标签出现频次 =====================
cfg.class_names = {'FMB','SawFM','MFT','ISRJ'};
cfg.class_num   = numel(cfg.class_names);

% (A) 单一干扰比例（你要“继承”到复合数据的标签出现频次）
%     例如 10:5:2:1
cfg.comp_ratio  = [10, 8, 6, 4];

% (B) 控制“复合程度”的关键参数：最常见成分的边际出现概率上限
%     p_max 越小，复合程度越低，四元组合越罕见。
%     推荐 0.35~0.6；若你希望四元极少，用 0.35~0.45
cfg.p_max = 0.55;

% (C) 数据集总样本数（全部为非空 multi-hot）
cfg.N_total = 12000;

% (D) 保存路径
cfg.save_data_path  = 'D:\干扰数据\长尾组合数据集生成\Jam4_ByMarginalRatio.mat';
cfg.save_label_path = 'D:\干扰数据\长尾组合数据集生成\label4_ByMarginalRatio.mat';

% (E) 边际精调：是否执行“计数微调”以让 sum(label) 更贴近目标
cfg.enable_count_repair = true;
cfg.max_repair_iters = 200000;

%% ===================== 2) 计算目标边际概率 m（满足比例） =====================
% 让最大频次成分的边际概率为 p_max，其余按比例缩放
r = cfg.comp_ratio(:);
m = cfg.p_max * (r / max(r));         % 目标：P(label_i=1 | sample) = m_i
m = min(max(m, 0.01), 0.95);          % 防极端
% ===== 可行性修正：必须保证 sum(m) > 1（非空多标签的必要条件）=====
sum_m = sum(m);
target_sum = 1.05;                 % 只需略大于 1，复合仍然很少
if sum_m <= 1
    scale = target_sum / sum_m;
    m = m * scale;
    m = min(m, 0.95);              % 防止过大
end

%% ===================== 3) 根据边际概率生成 multi-hot 模式分配 =====================
[patterns, counts] = allocate_pattern_counts(cfg.N_total, m, cfg.enable_count_repair, cfg.max_repair_iters);

% 生成 label_all
label_all = zeros(cfg.N_total, cfg.class_num);
pos = 1;
for j = 1:size(patterns,1)
    c = counts(j);
    if c <= 0, continue; end
    label_all(pos:pos+c-1, :) = repmat(patterns(j,:), c, 1);
    pos = pos + c;
end
label_all = label_all(randperm(size(label_all,1)), :);  % 打乱模式顺序

%% ===================== 4) 按 label_all 现场生成复合干扰（每条都来源于重新生成的单类并叠加） =====================
Jam_all = zeros(cfg.N_total, nPP);

for i = 1:cfg.N_total
    lab = label_all(i,:);
    sig = zeros(1, nPP);

    for compID = 1:cfg.class_num
        if lab(compID) == 1
            Jam_single = gen_single_jam(compID, 1, ...
                sig_LOS, nPulse, fs, nPP, nRepetition, flag_addTar, jam_delay_min, jam_delay_max, ...
                Bw, K, Tp, freq0, J_s_min, J_s_max, J_d_min, J_d_max);
            sig = sig + Jam_single;
        end
    end

    Jam_all(i,:) = sig;
end

%% ===================== 5) 拼接实虚部、打乱、打印统计、保存 =====================
data = [real(Jam_all), imag(Jam_all)];
label = label_all;

% perm = randperm(size(data,1));
% data  = data(perm,:);
% label = label(perm,:);

print_stats(label, cfg, m);

% save(cfg.save_data_path,  'data',  '-v7.3');
% save(cfg.save_label_path, 'label', '-v7.3');
% 
% fprintf('\n[Saved]\n  data : %s\n  label: %s\n', cfg.save_data_path, cfg.save_label_path);

end

%% =======================================================================
%% 生成单类干扰（与你之前一致：按 compID 调用对应 Data_gen_*）
function Jam = gen_single_jam(compID, nBatch_i, ...
    sig_LOS, nPulse, fs, nPP, nRepetition, flag_addTar, jam_delay_min, jam_delay_max, ...
    Bw, K, Tp, freq0, J_s_min, J_s_max, J_d_min, J_d_max)

sig_bandwidth = Bw;

switch compID
    case 1  % FMB 宽带压制
        JNR_min_FMB = J_s_min;  JNR_max_FMB = J_s_max;
        L_min_FMB = 2;  L_max_FMB = 5;
        Jam = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min_FMB, JNR_max_FMB, ...
            L_min_FMB, L_max_FMB, sig_bandwidth, freq0, fs, nPP, nBatch_i, nRepetition, ...
            flag_addTar, jam_delay_min, jam_delay_max);

    case 2  % SawFM 锯齿扫频
        JNR_min_SawFM = J_s_min;  JNR_max_SawFM = J_s_max;
        K_saw = 2*K;
        F_sweep_min_SawFM = 5;
        F_sweep_max_SawFM = 2;
        Jam = Data_gen_JsawFM(sig_LOS, JNR_min_SawFM, JNR_max_SawFM, Bw, K_saw, ...
            F_sweep_min_SawFM, F_sweep_max_SawFM, jam_delay_min, jam_delay_max, fs, ...
            nRepetition, nPulse, nBatch_i, flag_addTar);

    case 3  % MFT 密集假目标
        n_fake_tar_min_mt = 3;
        n_fake_tar_max_mt = 8;
        amp_min_mt = 20*log10(J_d_min);
        amp_max_mt = 20*log10(J_d_max);
        delay_min = 30e-6;
        delay_max = 50e-6;
        Jam = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min_mt, n_fake_tar_max_mt, ...
            delay_min, delay_max, amp_min_mt, amp_max_mt, nBatch_i, nRepetition, ...
            flag_addTar, jam_delay_min, jam_delay_max);

    case 4  % ISRJ 间歇采样转发
        JNR_min_ISRJ = J_d_min;  JNR_max_ISRJ = J_d_max;
        jam_tau_min_ISRJ = 5e-6;
        jam_tau_max_ISRJ = 20e-6;
        jam_replica_min_ISRJ = 1;
        jam_replica_max_ISRJ = 5;
        ISRJ_conv = 0;
        Jam = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min_ISRJ, JNR_max_ISRJ, ...
            jam_tau_min_ISRJ, jam_tau_max_ISRJ, jam_replica_min_ISRJ, jam_replica_max_ISRJ, ...
            ISRJ_conv, nBatch_i, nRepetition, flag_addTar, jam_delay_min, jam_delay_max);

    otherwise
        error('Unknown compID=%d', compID);
end

end

%% =============================================================
