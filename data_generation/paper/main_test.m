clc,clear,close all;
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
% figure;
% [S_tar, f, t] = spectrogram(sig_tar, hamming(128), 120, nPP, fs,'center');
% imagesc(db(S_tar));
%%干扰参数

nPulse = 1;
nBatch = 128;
freq0 = 0;
nRepetition = 1;
flag_addTar = 0;
jam_delay_min = 10e-6;
jam_delay_max = 10e-6;
%%速度假目标
% fdp_min = 2e6;
% fdp_max = 2e6;
% amp_min = 10*log10(20);
% amp_max = 10*log10(20);
% Jam_vft = Data_gen_Vector_fake_target(sig_LOS, nPulse, fs,K,Tp, ...
%     fdp_min, fdp_max, amp_min, amp_max, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% figure()
% plot(real(Jam_vft(1,:)));
% Jam_vft_f = fft(Jam_vft,[],2);
% figure()
% plot(db(Jam_vft_f(1,:)));
% for i = 1:nBatch
%     Jam_vft_r(i,:) = range_process_mf(Jam_vft(i,:).',sig_LOS.',nPP);
% end
% figure()
% plot(db(ifftshift(Jam_vft_r(1,:))));
% 
% S1 = fftshift(fft(Jam_vft_r,nBatch,1));
% figure;
% mesh(db(S1));
%% ISRJ
% JNR_min = 20;
% JNR_max = 20;
% jam_tau_min = 15e-6;
% jam_tau_max = 15e-6;
% jam_replica_min = 5;
% jam_replica_max = 5;
% ISRJ_conv = 0;
% Jam_ISRJ = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min, JNR_max, ...
%     jam_tau_min, jam_tau_max, jam_replica_min, jam_replica_max,ISRJ_conv, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% a = Jam_ISRJ;
% figure()
% plot(real(a(1,:)));
% a_f = fft(a,[],2);
% figure()
% plot(db(a_f(1,:)));
% for i = 1:nBatch
%     a_r(i,:) = range_process_mf(a(i,:).',sig_LOS.',nPP);
% end
% figure()
% plot(db(ifftshift(a_r(1,:))));
% 
% S1 = fftshift(fft(a_r,nBatch,1));
% figure;
% mesh(db(S1));
%% FMB
% JNR_min_FMB = 20;
% JNR_max_FMB = 40;
% L_min_FMB = 2;
% L_max_FMB = 5;
% sig_bandwidth = Bw;
% Jam_FMB = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min_FMB, JNR_max_FMB, ...
%     L_min_FMB, L_max_FMB, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%% AMA
% JNR_min_AMA = 20;
% JNR_max_AMA = 40;
% L_min_AMA = 1;
% L_max_AMA = 2;
% Jam_AMA = Data_gen_NoiseAM_Spot(sig_LOS, nPulse, JNR_min_AMA, JNR_max_AMA, ...
%     L_min_AMA, L_max_AMA, Bw, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%% SawFM
% JNR_min_SawFM = 20;
% JNR_max_SawFM = 40;
% K_saw = 2*K;
% F_sweep_min_SawFM = 5;
% F_sweep_max_SawFM = 5;
% Jam_SawFM = Data_gen_JsawFM(sig_LOS, JNR_min_SawFM, JNR_max_SawFM, Bw,K_saw, ...
%     F_sweep_min_SawFM, F_sweep_max_SawFM , jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
%%SinFM
% JNR_min_SinFM = 20;
% JNR_max_SinFM = 40;
% K_SinFM = 2*K;
% F_sweep_min_SinFM = 0.5;
% F_sweep_max_SinFM = 0.5;
% freq_jam = 1e6;
% fsin = 1e4;
% Jam_SinFM = Data_gen_SinFM(sig_LOS, JNR_min_SinFM, JNR_max_SinFM, Bw, freq_jam, fsin, ...
%     F_sweep_min_SinFM, F_sweep_max_SinFM, jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
%%灵巧
% JNR_min_SN = 20;
% JNR_max_SN = 40;
% jam_tau_min_SN = 5e-6;
% jam_tau_max_SN = 20e-6;
% jam_replica_min_SN = 1;
% jam_replica_max_SN = 4;
% Jam_SN = Data_gen_Smart_noise(sig_LOS, nPulse, fs, JNR_min_SN, JNR_max_SN, ...
%     jam_tau_min_SN, jam_tau_max_SN, jam_replica_min_SN, jam_replica_max_SN, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%密集
n_fake_tar_min_mt = 3;
n_fake_tar_max_mt = 8;
amp_min_mt = 20*log10(20);
amp_max_mt = 20*log10(40);
delay_min = 30e-6;
delay_max = 50e-6;
Jam_MFT = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min_mt, n_fake_tar_max_mt, ...
    delay_min, delay_max, amp_min_mt, amp_max_mt, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%速度假目标
% fdp_min_vft = 1e6;
% fdp_max_vft = 3e6;
% amp_min_vft = 20*log10(20);
% amp_max_vft = 20*log10(40);
% Jam_vft = Data_gen_Vector_fake_target(sig_LOS, nPulse, fs,K,Tp, ...
%     fdp_min_vft, fdp_max_vft, amp_min_vft, amp_max_vft, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);

%%间歇假目标
JNR_min_ISRJ = 20;
JNR_max_ISRJ = 40;
jam_tau_min_ISRJ = 5e-6;
jam_tau_max_ISRJ = 20e-6;
jam_replica_min_ISRJ = 1;
jam_replica_max_ISRJ = 5;
ISRJ_conv = 0;
Jam_ISRJ = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min_ISRJ, JNR_max_ISRJ, ...
    jam_tau_min_ISRJ, jam_tau_max_ISRJ, jam_replica_min_ISRJ, jam_replica_max_ISRJ,ISRJ_conv, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);

a = Jam_ISRJ;
figure()
plot(real(a(1,:)));
a_f = fft(a,[],2);
figure()
plot(db(a_f(1,:)));
for i = 1:nBatch
    a_r(i,:) = range_process_mf(a(i,:).',sig_LOS.',nPP);
end
figure()
plot(db(ifftshift(a_r(1,:))));
figure()
[S, f, t] = spectrogram(a(1,:), hamming(128), 120, nPP, fs,'center');
% [S, f, t] = spectrogram(a(1,:), hamming(4), 2, nPP, fs,'center');
imagesc(db(S));

S1 = fftshift(fft(a_r,nBatch,1));
figure;
mesh(db(S1));

%%123
% Jam_12 = Jam_FMB + Jam_AMA;
% Jam_13 = Jam_FMB + Jam_SawFM;
% Jam_14 = Jam_FMB + Jam_SinFM;
% Jam_15 = Jam_FMB + Jam_SN;
% Jam_16 = Jam_FMB + Jam_MFT;
% Jam_17 = Jam_FMB + Jam_vft;
% Jam_18 = Jam_FMB + Jam_ISRJ;
% 
% Jam_23 = Jam_AMA + Jam_SawFM;
% Jam_24 = Jam_AMA + Jam_SinFM;
% Jam_25 = Jam_AMA + Jam_SN;
% Jam_26 = Jam_AMA + Jam_MFT;
% Jam_27 = Jam_AMA + Jam_vft;
% Jam_28 = Jam_AMA + Jam_ISRJ;
% 
% Jam_34 = Jam_SawFM + Jam_SinFM;
% Jam_35 = Jam_SawFM + Jam_SN;
% Jam_36 = Jam_SawFM + Jam_MFT;
% Jam_37 = Jam_SawFM + Jam_vft;
% Jam_38 = Jam_SawFM + Jam_ISRJ;
% 
% Jam_45 = Jam_SinFM + Jam_SN;
% Jam_46 = Jam_SinFM + Jam_MFT;
% Jam_47 = Jam_SinFM + Jam_vft;
% Jam_48 = Jam_SinFM + Jam_ISRJ;
% 
% Jam_56 = Jam_SN + Jam_MFT;
% Jam_57 = Jam_SN+ Jam_vft;
% Jam_58 = Jam_SN + Jam_ISRJ;
% 
% Jam_67 = Jam_MFT + Jam_vft;
% Jam_68 = Jam_MFT + Jam_ISRJ;
% 
% Jam_78 = Jam_vft + Jam_ISRJ;
% 
% Jam_123 = Jam_FMB + Jam_AMA + Jam_SawFM;
% Jam_124 = Jam_FMB + Jam_AMA + Jam_SinFM;
% Jam_125 = Jam_FMB + Jam_AMA + Jam_SN;
% Jam_126 = Jam_FMB + Jam_AMA + Jam_MFT;
% Jam_127 = Jam_FMB + Jam_AMA + Jam_vft;
% Jam_128 = Jam_FMB + Jam_AMA + Jam_ISRJ;
% 
% Jam_134 = Jam_FMB + Jam_SawFM + Jam_SinFM;
% Jam_135 = Jam_FMB + Jam_SawFM + Jam_SN;
% Jam_136 = Jam_FMB + Jam_SawFM + Jam_MFT;
% Jam_137 = Jam_FMB + Jam_SawFM + Jam_vft;
% Jam_138 = Jam_FMB + Jam_SawFM + Jam_ISRJ;
% 
% Jam_145 = Jam_FMB + Jam_SawFM + Jam_SN;
% Jam_146 = Jam_FMB + Jam_SawFM + Jam_MFT;
% Jam_147 = Jam_FMB + Jam_SawFM + Jam_vft;
% Jam_148 = Jam_FMB + Jam_SawFM + Jam_ISRJ;

