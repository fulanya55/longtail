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
% 长尾：按组合大小分配样本数（建议可调）
N_by_size = zeros(1, class_num);
N_by_size(1) = 400;  % 单一干扰（Head）
N_by_size(2) = 200;  % 两种干扰复合
N_by_size(3) = 80;   % 三种干扰复合
N_by_size(4) = 30;   % 四种干扰复合（Tail）
% 长尾指数可以调整
% N_head = 400;
% decay = 0.5;        % 越小越长尾
% N_min = 20;
% N_by_size = max(round(N_head * decay.^(0:class_num-1)), N_min);

Nmax = max(N_by_size);    % 原子库每类生成 Nmax 条
nBatch = Nmax;            % 让你原来的各个 Data_gen_* 都生成 Nmax
%% 干扰参数设置

%参数组合1
J_s_min = 30;
J_s_max = 50;
J_d_min = 20;
J_d_max = 40;

%%单一类型干扰产生
%%宽带压制
JNR_min_FMB = J_s_min;
JNR_max_FMB = J_s_max;
L_min_FMB = 2;
L_max_FMB = 5;
sig_bandwidth = Bw;
Jam_FMB = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min_FMB, JNR_max_FMB, ...
    L_min_FMB, L_max_FMB, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%调幅瞄准
JNR_min_AMA = J_s_min;
JNR_max_AMA = J_s_max;
L_min_AMA = 1;
L_max_AMA = 2;
Jam_AMA = Data_gen_NoiseAM_Spot(sig_LOS, nPulse, JNR_min_AMA, JNR_max_AMA, ...
    L_min_AMA, L_max_AMA, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%锯齿形函数扫频
JNR_min_SawFM = J_s_min;
JNR_max_SawFM = J_s_max;
K_saw = 2*K;
F_sweep_min_SawFM = 5;
F_sweep_max_SawFM = 2;
Jam_SawFM = Data_gen_JsawFM(sig_LOS, JNR_min_SawFM, JNR_max_SawFM, Bw,K_saw, ...
    F_sweep_min_SawFM, F_sweep_max_SawFM , jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
%%正弦函数扫频
JNR_min_SinFM = J_s_min;
JNR_max_SinFM = J_s_max;
F_sweep_min_SinFM = 0.5;
F_sweep_max_SinFM = 1;
freq_jam = 0.5e6+1e6*rand();
fsin = 1e4+1e4*rand();%1e3-1e5
Jam_SinFM = Data_gen_SinFM(sig_LOS, JNR_min_SinFM, JNR_max_SinFM, Bw, freq_jam, fsin, ...
    F_sweep_min_SinFM, F_sweep_max_SinFM, jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
%%灵巧噪声干扰
JNR_min_SN = J_s_min;
JNR_max_SN = J_s_max;
jam_tau_min_SN = 5e-6;
jam_tau_max_SN = 20e-6;
jam_replica_min_SN = 1;
jam_replica_max_SN = 4;
Jam_SN = Data_gen_Smart_noise(sig_LOS, nPulse, fs, JNR_min_SN, JNR_max_SN, ...
    jam_tau_min_SN, jam_tau_max_SN, jam_replica_min_SN, jam_replica_max_SN, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%密集假目标干扰
n_fake_tar_min_mt = 3;
n_fake_tar_max_mt = 8;
amp_min_mt = 20*log10(J_d_min);
amp_max_mt = 20*log10(J_d_max);
delay_min = 30e-6;
delay_max = 50e-6;
Jam_MFT = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min_mt, n_fake_tar_max_mt, ...
    delay_min, delay_max, amp_min_mt, amp_max_mt, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%速度假目标干扰
fdp_min_vft = 1e6;
fdp_max_vft = 3e6;
amp_min_vft = 20*log10(J_d_min);
amp_max_vft = 20*log10(J_d_max);
Jam_vft = Data_gen_Vector_fake_target(sig_LOS, nPulse, fs,K,Tp, ...
    fdp_min_vft, fdp_max_vft, amp_min_vft, amp_max_vft, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%间歇采样转发干扰
JNR_min_ISRJ = J_d_min;
JNR_max_ISRJ = J_d_max;
jam_tau_min_ISRJ = 5e-6;
jam_tau_max_ISRJ = 20e-6;
jam_replica_min_ISRJ = 1;
jam_replica_max_ISRJ = 5;
ISRJ_conv = 0;
Jam_ISRJ = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min_ISRJ, JNR_max_ISRJ, ...
    jam_tau_min_ISRJ, jam_tau_max_ISRJ, jam_replica_min_ISRJ, jam_replica_max_ISRJ,ISRJ_conv, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%% 单类汇总数据
% Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_SinFM,Jam_SN,Jam_MFT,Jam_vft,Jam_ISRJ};
% Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_SinFM,Jam_MFT,Jam_vft,Jam_ISRJ};
Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_MFT,Jam_ISRJ};
arrays = Jam_all_sigal;
%% ==================== 3) 生成长尾复合干扰数据集 ====================
Jam_all = [];
label_all = [];

for k = 1:length(arrays)   % k = 组合大小（1~4）
    curN = N_by_size(k);   % 当前组合生成样本数（长尾）

    combinationIndices = nchoosek(1:length(arrays), k);

    for i = 1:size(combinationIndices, 1)
        selected = combinationIndices(i, :);

        % ---- 生成 curN 条该组合的复合干扰 ----
        Jam_tem = zeros(curN, nPP);

        for j = 1:length(selected)
            % 为每个成分随机抽取 curN 条（增强多样性）
            idx = randi(Nmax, curN, 1);
            Jam_tem = Jam_tem + arrays{selected(j)}(idx, :);
        end

        % ---- 生成 multi-hot 标签（长度=class_num）----
        lab = zeros(1, class_num);
        lab(selected) = 1;      % 选中的成分置 1
        label_tem = repmat(lab, curN, 1);

        % ---- 汇总 ----
        Jam_all   = [Jam_all; Jam_tem];
        label_all = [label_all; label_tem];
    end
end

%% ==================== 4) 复数拼接与保存 ====================
% 原来你是 [real, imag] 拼接
% data = [real(Jam_all), imag(Jam_all)];
data = Jam_all;
label = label_all;

save('D:\干扰数据\长尾组合数据集\Jam4_LongTail_1.mat','data','-v7.3');
save('D:\干扰数据\长尾组合数据集\label4_LongTail_1.mat','label','-v7.3');









