clc,clear,close all;
%%生成长尾数据集
%% 信号参数
rng('shuffle'); 
C = 3e8;
Tp = 100e-6;     %脉宽
Tpp = 300e-6;    %脉冲重复周期
duty_cycle = Tp/Tpp; %占空比
% Tpp = 100e-6;    %脉冲重复周期
fs = 10e6;      %采样率
Bw = 5e6;      %带宽
Np = Tp * fs;     %脉冲内点数
nPP = ceil(Tpp * fs);   %脉冲重复周期内点数
K = Bw / Tp;      %调频斜率
% tp = (-Np/2 : Np/2-1)' / fs;
tp = (1 : Np)' / fs;
% t = (-nPP/2: nPP/2-1)/fs;
t = (1: nPP)/fs;
R = t*C/2;
R_max = Tpp*C/2;
R_det = C/2/Bw;
%% 本振信号
sigLFM = exp(1i*pi*K*tp.^2);  %线性调频信号
SigLFM = abs(fftshift(fft(sigLFM)));
sig_LOS = [sigLFM ; zeros(ceil(nPP-Np), 1)];  %单脉冲信号
%% 目标信号
SNR = 30;
tar_range = 10000;
tar_delay = tar_range/2/C;
tar_dp = round(tar_delay*fs);
sig_tar = circshift(sig_LOS,tar_dp);
sig_tar = awgn(sig_tar,SNR);
% figure()
% plot(t,real(sig_tar));
% Sig_tar = fftshift(range_process_mf(sig_tar,sig_LOS.',nPP));
% figure()
% plot(R,db(Sig_tar));
%% 压制干扰信号
%压制干扰包括宽带压制、窄带瞄准、杂乱脉冲、函数扫频（锯齿、正弦）
%%干扰参数
nPulse = 1;
% nBatch = 400;
freq0 = 0;
nRepetition = 1;
flag_addTar = 0;
jam_delay_min = 10e-6;
jam_delay_max = 50e-6;
%% ==================== 1) 长尾参数设置 ====================
% 4 类基础干扰：FMB, SawFM, MFT, ISRJ
% 标签均衡，组合数据不均衡
class_num = 4;%与后面的干扰数量对应
%% ==================== 长尾参数设置 ====================
class_num = 4;  % {FMB, SawFM, MFT, ISRJ}

% (1) 每个成分的“单类样本库”大小（不均衡）
% 你可按需要改成更陡的长尾，比如 [1000 500 200 80]
N_pool = [800, 500, 250, 120];

% (2) 组合大小对应的“基础生成数量”（长尾）
% 单一/二元/三元/四元
N_by_size = [400, 200, 80, 30];

% (3) 可选：是否启用“组合稀有度缩放”
use_combo_rarity_scaling = true;
eta = 1.0;              % 缩放强度，越大尾部越少（0=不缩放）
N_min_per_combo = 10;   % 每个组合最少样本数，避免太少
%% 干扰参数设置

%参数组合1
J_s_min = 30;
J_s_max = 50;
J_d_min = 20;
J_d_max = 40;

%%单一类型干扰产生
% %%宽带压制
% JNR_min_FMB = J_s_min;
% JNR_max_FMB = J_s_max;
% L_min_FMB = 2;
% L_max_FMB = 5;
% sig_bandwidth = Bw;
% Jam_FMB = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min_FMB, JNR_max_FMB, ...
%     L_min_FMB, L_max_FMB, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% %%调幅瞄准
% JNR_min_AMA = J_s_min;
% JNR_max_AMA = J_s_max;
% L_min_AMA = 1;
% L_max_AMA = 2;
% Jam_AMA = Data_gen_NoiseAM_Spot(sig_LOS, nPulse, JNR_min_AMA, JNR_max_AMA, ...
%     L_min_AMA, L_max_AMA, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% %%锯齿形函数扫频
% JNR_min_SawFM = J_s_min;
% JNR_max_SawFM = J_s_max;
% K_saw = 2*K;
% F_sweep_min_SawFM = 5;
% F_sweep_max_SawFM = 2;
% Jam_SawFM = Data_gen_JsawFM(sig_LOS, JNR_min_SawFM, JNR_max_SawFM, Bw,K_saw, ...
%     F_sweep_min_SawFM, F_sweep_max_SawFM , jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
% %%正弦函数扫频
% JNR_min_SinFM = J_s_min;
% JNR_max_SinFM = J_s_max;
% F_sweep_min_SinFM = 0.5;
% F_sweep_max_SinFM = 1;
% freq_jam = 0.5e6+1e6*rand();
% fsin = 1e4+1e4*rand();%1e3-1e5
% Jam_SinFM = Data_gen_SinFM(sig_LOS, JNR_min_SinFM, JNR_max_SinFM, Bw, freq_jam, fsin, ...
%     F_sweep_min_SinFM, F_sweep_max_SinFM, jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
% %%灵巧噪声干扰
% JNR_min_SN = J_s_min;
% JNR_max_SN = J_s_max;
% jam_tau_min_SN = 5e-6;
% jam_tau_max_SN = 20e-6;
% jam_replica_min_SN = 1;
% jam_replica_max_SN = 4;
% Jam_SN = Data_gen_Smart_noise(sig_LOS, nPulse, fs, JNR_min_SN, JNR_max_SN, ...
%     jam_tau_min_SN, jam_tau_max_SN, jam_replica_min_SN, jam_replica_max_SN, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% %%密集假目标干扰
% n_fake_tar_min_mt = 3;
% n_fake_tar_max_mt = 8;
% amp_min_mt = 20*log10(J_d_min);
% amp_max_mt = 20*log10(J_d_max);
% delay_min = 30e-6;
% delay_max = 50e-6;
% Jam_MFT = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min_mt, n_fake_tar_max_mt, ...
%     delay_min, delay_max, amp_min_mt, amp_max_mt, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% %%速度假目标干扰
% fdp_min_vft = 1e6;
% fdp_max_vft = 3e6;
% amp_min_vft = 20*log10(J_d_min);
% amp_max_vft = 20*log10(J_d_max);
% Jam_vft = Data_gen_Vector_fake_target(sig_LOS, nPulse, fs,K,Tp, ...
%     fdp_min_vft, fdp_max_vft, amp_min_vft, amp_max_vft, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% %%间歇采样转发干扰
% JNR_min_ISRJ = J_d_min;
% JNR_max_ISRJ = J_d_max;
% jam_tau_min_ISRJ = 5e-6;
% jam_tau_max_ISRJ = 20e-6;
% jam_replica_min_ISRJ = 1;
% jam_replica_max_ISRJ = 5;
% ISRJ_conv = 0;
% Jam_ISRJ = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min_ISRJ, JNR_max_ISRJ, ...
%     jam_tau_min_ISRJ, jam_tau_max_ISRJ, jam_replica_min_ISRJ, jam_replica_max_ISRJ,ISRJ_conv, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% %% 单类汇总数据
% % Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_SinFM,Jam_SN,Jam_MFT,Jam_vft,Jam_ISRJ};
% % Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_SinFM,Jam_MFT,Jam_vft,Jam_ISRJ};
% Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_MFT,Jam_ISRJ};
% arrays = Jam_all_sigal;
% ---------- 1) FMB ----------
nBatch_i = N_pool(1);
JNR_min_FMB = J_s_min; JNR_max_FMB = J_s_max;
L_min_FMB = 2; L_max_FMB = 5;
sig_bandwidth = Bw;
Jam_FMB = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min_FMB, JNR_max_FMB, ...
    L_min_FMB, L_max_FMB, sig_bandwidth, freq0, fs, nPP, nBatch_i, nRepetition, ...
    flag_addTar , jam_delay_min, jam_delay_max);

% ---------- 2) SawFM ----------
nBatch_i = N_pool(2);
JNR_min_SawFM = J_s_min; JNR_max_SawFM = J_s_max;
K_saw = 2*K;
F_sweep_min_SawFM = 5;
F_sweep_max_SawFM = 2;
Jam_SawFM = Data_gen_JsawFM(sig_LOS, JNR_min_SawFM, JNR_max_SawFM, Bw, K_saw, ...
    F_sweep_min_SawFM, F_sweep_max_SawFM , jam_delay_min, jam_delay_max, fs, ...
    nRepetition, nPulse, nBatch_i, flag_addTar);

% ---------- 3) MFT ----------
nBatch_i = N_pool(3);
n_fake_tar_min_mt = 3; n_fake_tar_max_mt = 8;
amp_min_mt = 20*log10(J_d_min); amp_max_mt = 20*log10(J_d_max);
delay_min = 30e-6; delay_max = 50e-6;
Jam_MFT = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min_mt, n_fake_tar_max_mt, ...
    delay_min, delay_max, amp_min_mt, amp_max_mt, nBatch_i, nRepetition, ...
    flag_addTar , jam_delay_min, jam_delay_max);

% ---------- 4) ISRJ ----------
nBatch_i = N_pool(4);
JNR_min_ISRJ = J_d_min; JNR_max_ISRJ = J_d_max;
jam_tau_min_ISRJ = 5e-6; jam_tau_max_ISRJ = 20e-6;
jam_replica_min_ISRJ = 1; jam_replica_max_ISRJ = 5;
ISRJ_conv = 0;
Jam_ISRJ = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min_ISRJ, JNR_max_ISRJ, ...
    jam_tau_min_ISRJ, jam_tau_max_ISRJ, jam_replica_min_ISRJ, jam_replica_max_ISRJ, ...
    ISRJ_conv, nBatch_i, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);

% 汇总成 cell（注意顺序必须与 class_num 对应）
arrays = {Jam_FMB, Jam_SawFM, Jam_MFT, Jam_ISRJ};
%% ==================== 生成长尾复合干扰数据集 ====================
Jam_all = [];
label_all = [];

N_pool_max = max(N_pool);

for k = 1:class_num  % k = 组合大小（1~4）
    baseN = N_by_size(k);
    combinationIndices = nchoosek(1:class_num, k);

    for i = 1:size(combinationIndices, 1)
        selected = combinationIndices(i, :);

        % ---- (可选) 稀有度缩放：组合里包含越稀有的成分，生成数量越少 ----
        if use_combo_rarity_scaling
            rarityFactor = min(N_pool(selected)) / N_pool_max;  % 0~1
            curN = max(round(baseN * (rarityFactor^eta)), N_min_per_combo);
        else
            curN = baseN;
        end

        % ---- 生成 curN 条该组合的复合干扰 ----
        Jam_tem = zeros(curN, nPP);

        for j = 1:length(selected)
            compID = selected(j);
            Ni = size(arrays{compID}, 1);     % 该成分库大小（不均衡）
            idx = randi(Ni, curN, 1);         % 带放回抽样，保证不会越界
            Jam_tem = Jam_tem + arrays{compID}(idx, :);
        end

        % ---- multi-hot 标签 ----
        lab = zeros(1, class_num);
        lab(selected) = 1;
        label_tem = repmat(lab, curN, 1);

        Jam_all   = [Jam_all; Jam_tem];
        label_all = [label_all; label_tem];
    end
end

%% ==================== 4) 复数拼接与保存 ====================
% 原来你是 [real, imag] 拼接
data = [real(Jam_all), imag(Jam_all)];
label = label_all;

% perm = randperm(size(data,1));
% data  = data(perm,:);
% label = label(perm,:);
fprintf('\n==================== Dataset Statistics ====================\n');
fprintf('Total samples: %d\n', size(label_all,1));

% 1) 单类干扰数量（每个成分在样本中出现的边际次数）
%    注意：这是“成分被标 1 的次数”，不是“单一干扰样本数”
marginal_pos = sum(label_all, 1);
fprintf('\n[1] Marginal positives by component (label=1 count):\n');
for c = 1:class_num
    fprintf('  Component-%d : %d\n', c, marginal_pos(c));
end

% 1.1) 单一干扰样本数（严格意义的“单类样本”= k=1 且该成分为1）
fprintf('\n[1.1] Pure single-component samples (k=1) by component:\n');
k_vec = sum(label_all, 2);  % 每条样本的成分数
fprintf('\n[1.1] Pure single-component samples (k=1) by component:\n');
fprintf('\n[1.1] Pure single-component samples (k=1) by component:\n');
for c = 1:class_num
    cnt_single_c = sum((k_vec==1) & (label_all(:,c)==1));
    pat = zeros(1, class_num);
    pat(c) = 1;
    fprintf('  Pattern %s : %d\n', mat2str(pat), cnt_single_c);
end

% 2) 干扰成分数量分布（k=1/2/3/4）
fprintf('\n[2] Samples by number of active components k:\n');
for k = 1:class_num
    fprintf('  k=%d : %d\n', k, sum(k_vec==k));
end

% 3) 每种组合干扰数量（multi-hot 模式统计）
fprintf('\n[3] Counts of each multi-hot pattern:\n');
[uniq_rows, ~, ic] = unique(label_all, 'rows');
cnt = accumarray(ic, 1);

% 按数量降序排序输出
[sorted_cnt, idx] = sort(cnt, 'descend');
for i = 1:numel(sorted_cnt)
    pat = uniq_rows(idx(i), :);
    fprintf('  %s : %d\n', mat2str(pat), sorted_cnt(i));
end
fprintf('=============================================================\n\n');
% save('D:\干扰数据\复合干扰数据\Jam4_LongTail_ComponentImb.mat','data','-v7.3');
% save('D:\干扰数据\复合干扰数据\label4_LongTail_ComponentImb.mat','label','-v7.3');








