function gen_longtail_composite_dataset()
clc; clear; close all;

%% ===================== 0) 你的原始参数段（保持不变） =====================
rng('shuffle');

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

% 目标信号参数（此处未用，可保留）
SNR = 30;
tar_range = 10000;

% 干扰全局参数
nPulse = 1;
nRepetition = 1;
flag_addTar = 0;
freq0 = 0;
jam_delay_min = 10e-6;
jam_delay_max = 50e-6;

% 你原来用到的功率参数（务必给出具体数值）
% 下面仅占位：请按你的实验设定填写
J_s_min = 30;  J_s_max = 50;
J_d_min = 20;  J_d_max = 40;

%% ===================== 1) 配置：成分不均衡 + 组合长尾 =====================
cfg.class_names = {'FMB','SawFM','MFT','ISRJ'};
cfg.class_num   = numel(cfg.class_names);   % =4

% (A) 成分不均衡比例（越大越常见）
%     对应 [FMB SawFM MFT ISRJ]
cfg.comp_ratio = [10, 8, 6, 4];

% (B) 组合长尾比例（按组合大小 1/2/3/4 元）
cfg.size_ratio = [10, 4, 2, 1];

% (C) 总体规模：按“1元组合总数”设定基准，然后按 size_ratio 推其他
%     你可以把 N1_total 设大一些（例如 20000）生成更大数据集
cfg.N1_total = 8000; % 所有“单一干扰样本”总数（不是每个类）

% (D) 是否对“包含稀有成分的组合”做额外缩放（更像真实长尾）
cfg.use_combo_rarity_scaling = true;
cfg.rarity_eta = 1.0;         % 缩放强度，越大尾部组合越少
cfg.min_per_combo = 50;       % 每个具体组合最少样本数（避免极端过小）

% (E) 保存路径
cfg.save_data_path  = 'D:\干扰数据\复合干扰数据\Jam4_LongTail_Final.mat';
cfg.save_label_path = 'D:\干扰数据\复合干扰数据\label4_LongTail_Final.mat';

%% ===================== 2) 计算各组合大小总样本数 =====================
% 按 size_ratio 计算 N_k_total
size_ratio = cfg.size_ratio(:);
Nk_total = round(cfg.N1_total * (size_ratio / size_ratio(1)));  % 以 N1_total 为基准
% Nk_total(1)=N1_total, Nk_total(2..4)按比例缩放
fprintf('\n[Config] Total samples by combo size k=1..%d:\n', cfg.class_num);
disp(Nk_total.');

%% ===================== 3) 为每个“具体组合模式”分配样本数 =====================
% 组合模式：用 multi-hot 作为 key，例如 [1 0 1 0] 表示 FMB+MFT
combo_list = cell(cfg.class_num,1);
combo_counts = cell(cfg.class_num,1);

for k = 1:cfg.class_num
    combos = nchoosek(1:cfg.class_num, k);
    combo_list{k} = combos;

    % 给每个具体组合一个权重：越稀有成分越小（可选）
    if cfg.use_combo_rarity_scaling
        w = zeros(size(combos,1),1);
        for i = 1:size(combos,1)
            sel = combos(i,:);
            % 稀有度：用 comp_ratio 的最小值（包含最稀有成分则权重小）
            rarityFactor = min(cfg.comp_ratio(sel)) / max(cfg.comp_ratio);
            w(i) = rarityFactor ^ cfg.rarity_eta;
        end
        w = w / sum(w);
    else
        w = ones(size(combos,1),1) / size(combos,1);
    end

    % 分配到每个具体组合
    cnt = max(round(Nk_total(k) * w), cfg.min_per_combo);
    % 若总数偏离太多，可再做一次归一化回 Nk_total(k)（此处简单处理即可）
    combo_counts{k} = cnt;
end

%% ===================== 4) 逐组合“现场生成单类干扰并叠加” =====================
Jam_all = [];
label_all = [];

for k = 1:cfg.class_num
    combos = combo_list{k};
    cnts   = combo_counts{k};

    fprintf('\n[Generate] k=%d, #patterns=%d\n', k, size(combos,1));

    for i = 1:size(combos,1)
        selected = combos(i,:);
        curN = cnts(i);

        % multi-hot label
        lab = zeros(1,cfg.class_num);
        lab(selected) = 1;

        % 现场生成并叠加
        Jam_tem = zeros(curN, nPP);

        for j = 1:numel(selected)
            compID = selected(j);
            Jam_single = gen_single_jam(compID, curN, ...
                sig_LOS, nPulse, fs, nPP, nRepetition, flag_addTar, jam_delay_min, jam_delay_max, ...
                Bw, K, Tp, freq0, J_s_min, J_s_max, J_d_min, J_d_max);

            Jam_tem = Jam_tem + Jam_single;
        end

        Jam_all   = [Jam_all; Jam_tem];
        label_all = [label_all; repmat(lab, curN, 1)];
    end
end

%% ===================== 5) 拼接实虚部、打乱、统计打印、保存 =====================
data = [real(Jam_all), imag(Jam_all)];
label = label_all;

% perm = randperm(size(data,1));
% data  = data(perm,:);
% label = label(perm,:);

print_dataset_stats(label, cfg);

% save(cfg.save_data_path,  'data',  '-v7.3');
% save(cfg.save_label_path, 'label', '-v7.3');
% 
% fprintf('\n[Saved]\n  data : %s\n  label: %s\n', cfg.save_data_path, cfg.save_label_path);
end

%% ========================================================================
%% 单类干扰生成（按你的四类：FMB / SawFM / MFT / ISRJ）
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

%% ========================================================================
%% 打印统计：组合大小、具体组合、成分边际次数
function print_dataset_stats(label, cfg)
N = size(label,1);
k = sum(label,2);

fprintf('\n[Stats] Total samples: %d\n', N);

% (1) 按组合大小统计
fprintf('[Stats] Samples by combo size (#active labels):\n');
for kk = 1:cfg.class_num
    fprintf('  k=%d : %d\n', kk, sum(k==kk));
end

% (2) 成分边际正样本次数（每个成分被标1的次数）
fprintf('[Stats] Marginal positives by component:\n');
marg = sum(label,1);
for i = 1:cfg.class_num
    fprintf('  %-6s : %d\n', cfg.class_names{i}, marg(i));
end

% (3) 具体组合模式统计（multi-hot 模式）
fprintf('[Stats] Top combination patterns (multi-hot):\n');
[uniq_rows,~,ic] = unique(label, 'rows');
cnt = accumarray(ic,1);
[sorted_cnt, idx] = sort(cnt, 'descend');
topK = min(20, numel(sorted_cnt));

for t = 1:topK
    pat = uniq_rows(idx(t),:);
    fprintf('  %s : %d\n', mat2str(pat), sorted_cnt(t));
end
end