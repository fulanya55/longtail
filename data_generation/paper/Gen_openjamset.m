clc,clear,close all;
% 根入口按自身位置解析依赖，运行时无需依赖外部 MATLAB 当前目录。
repoRoot = fileparts(mfilename('fullpath'));
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(fullfile(repoRoot, 'Jam9'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成', 'core'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成', 'generators'));
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
nBatch = 400;
freq0 = 0;
nRepetition = 1;
flag_addTar = 0;
jam_delay_min = 10e-6;
jam_delay_max = 50e-6;
%%单一类型干扰产生
%%宽带压制
JNR_min_FMB = 20;
JNR_max_FMB = 40;
L_min_FMB = 2;
L_max_FMB = 5;
sig_bandwidth = Bw;
[Jam_FMB, truth_FMB] = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min_FMB, JNR_max_FMB, ...
    L_min_FMB, L_max_FMB, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%调幅瞄准
JNR_min_AMA = 20;
JNR_max_AMA = 40;
L_min_AMA = 1;
L_max_AMA = 2;
[Jam_AMA, truth_AMA] = Data_gen_NoiseAM_Spot(sig_LOS, nPulse, JNR_min_AMA, JNR_max_AMA, ...
    L_min_AMA, L_max_AMA, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%锯齿形函数扫频
JNR_min_SawFM = 20;
JNR_max_SawFM = 40;
K_saw = 2*K;
F_sweep_min_SawFM = 5;
F_sweep_max_SawFM = 2;
Jam_SawFM = Data_gen_JsawFM(sig_LOS, JNR_min_SawFM, JNR_max_SawFM, Bw,K_saw, ...
    F_sweep_min_SawFM, F_sweep_max_SawFM , jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
%%正弦函数扫频
JNR_min_SinFM = 20;
JNR_max_SinFM = 40;
F_sweep_min_SinFM = 0.5;
F_sweep_max_SinFM = 1;
freq_jam = 0.5e6+1e6*rand();
fsin = 1e4+1e4*rand();%1e3-1e5
Jam_SinFM = Data_gen_SinFM(sig_LOS, JNR_min_SinFM, JNR_max_SinFM, Bw, freq_jam, fsin, ...
    F_sweep_min_SinFM, F_sweep_max_SinFM, jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar);
%%灵巧噪声干扰
JNR_min_SN = 20;
JNR_max_SN = 40;
jam_tau_min_SN = 5e-6;
jam_tau_max_SN = 20e-6;
jam_replica_min_SN = 1;
jam_replica_max_SN = 4;
Jam_SN = Data_gen_Smart_noise(sig_LOS, nPulse, fs, JNR_min_SN, JNR_max_SN, ...
    jam_tau_min_SN, jam_tau_max_SN, jam_replica_min_SN, jam_replica_max_SN, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%密集假目标干扰
n_fake_tar_min_mt = 3;
n_fake_tar_max_mt = 8;
amp_min_mt = 20*log10(20);
amp_max_mt = 20*log10(40);
delay_min = 30e-6;
delay_max = 50e-6;
[Jam_MFT, truth_MFT] = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min_mt, n_fake_tar_max_mt, ...
    delay_min, delay_max, amp_min_mt, amp_max_mt, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
atomic_truth = struct('FMB', truth_FMB, 'AMA', truth_AMA, ...
    'MFT', truth_MFT);
%%速度假目标干扰
fdp_min_vft = 1e6;
fdp_max_vft = 3e6;
amp_min_vft = 20*log10(20);
amp_max_vft = 20*log10(40);
Jam_vft = Data_gen_Vector_fake_target(sig_LOS, nPulse, fs,K,Tp, ...
    fdp_min_vft, fdp_max_vft, amp_min_vft, amp_max_vft, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
%%间歇采样转发干扰
JNR_min_ISRJ = 20;
JNR_max_ISRJ = 40;
jam_tau_min_ISRJ = 5e-6;
jam_tau_max_ISRJ = 20e-6;
jam_replica_min_ISRJ = 1;
jam_replica_max_ISRJ = 5;
ISRJ_conv = 0;
Jam_ISRJ = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min_ISRJ, JNR_max_ISRJ, ...
    jam_tau_min_ISRJ, jam_tau_max_ISRJ, jam_replica_min_ISRJ, jam_replica_max_ISRJ,ISRJ_conv, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);

%% 单类汇总数据
% 选用5种干扰已知，一种未知
% Jam_all_sigal = [Jam_FMB;Jam_AMA;Jam_SawFM;Jam_MFT;Jam_ISRJ];
%% 复合干扰生成
%FMB:1 AMA:2 SawFM:3  MFT:4  ISRJ:5
% 未知：SinFM SN Vft
%% 未知选用1，2345混合
Jam_all_sigal = [Jam_AMA;Jam_SawFM;Jam_MFT;Jam_ISRJ];
Jam_2 = [];
for i = 1:3
    for j = i+1:4
        Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
        Jam_2 = [Jam_2;Jam_tem];
    end
end
Jam_3 = [];
for i = 1:2
    for j = i+1:3
        for ii = j+1:4
            Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
            Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
            Jam_3 = [Jam_3;Jam_tem];
        end
    end
end
Jam_4 = zeros(nBatch,size(Jam_all_sigal,2));
for i = 1:4
    Jam_4 = Jam_4 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
end
Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4];
%制作标签
class_num = 4;
zhuhe_num = 4;
label = zeros(size(Jam_All,1),class_num);
zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
for i = 1:size(zuhe_one_hot,1)
    label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
end
Jam_all = [real(Jam_All),imag(Jam_All)];
data = Jam_all;
%% 保存
% 第一份开放集组合保留原主文件，并增加同 stem 的真值旁车。
legacy_class_codes_1 = {'AMA','SawFM','MFT','ISRJ'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
mainFilePath = 'Jam5_1.mat';
labelFilePath = 'label5_1.mat';
truthMetadata = struct('legacy_entrypoint', 'Gen_openjamset.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition, ...
    'open_set_partition_index', 1);
truthBundle = build_legacy_truth_bundle(label, legacy_class_codes_1, ...
    atomic_truth, source_sample_index, truthMetadata);
save(mainFilePath, 'data', '-v7.3');
save(labelFilePath, 'label', '-v7.3');
save_legacy_truth_sidecar(mainFilePath, truthBundle, true);

%% 未知选用2，1345混合
Jam_all_sigal = [Jam_FMB;Jam_SawFM;Jam_MFT;Jam_ISRJ];
Jam_2 = [];
for i = 1:3
    for j = i+1:4
        Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
        Jam_2 = [Jam_2;Jam_tem];
    end
end
Jam_3 = [];
for i = 1:2
    for j = i+1:3
        for ii = j+1:4
            Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
            Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
            Jam_3 = [Jam_3;Jam_tem];
        end
    end
end
Jam_4 = zeros(nBatch,size(Jam_all_sigal,2));
for i = 1:4
    Jam_4 = Jam_4 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
end
Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4];
%制作标签
class_num = 4;
zhuhe_num = 4;
label = zeros(size(Jam_All,1),class_num);
zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
for i = 1:size(zuhe_one_hot,1)
    label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
end
Jam_all = [real(Jam_All),imag(Jam_All)];
data = Jam_all;
%% 保存
% 第二份开放集组合使用显式类别列映射生成真值旁车。
legacy_class_codes_2 = {'FMB','SawFM','MFT','ISRJ'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
mainFilePath = 'Jam5_2.mat';
labelFilePath = 'label5_2.mat';
truthMetadata = struct('legacy_entrypoint', 'Gen_openjamset.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition, ...
    'open_set_partition_index', 2);
truthBundle = build_legacy_truth_bundle(label, legacy_class_codes_2, ...
    atomic_truth, source_sample_index, truthMetadata);
save(mainFilePath, 'data', '-v7.3');
save(labelFilePath, 'label', '-v7.3');
save_legacy_truth_sidecar(mainFilePath, truthBundle, true);

%% 未知选用3，1245混合
Jam_all_sigal = [Jam_FMB;Jam_AMA;Jam_MFT;Jam_ISRJ];
Jam_2 = [];
for i = 1:3
    for j = i+1:4
        Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
        Jam_2 = [Jam_2;Jam_tem];
    end
end
Jam_3 = [];
for i = 1:2
    for j = i+1:3
        for ii = j+1:4
            Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
            Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
            Jam_3 = [Jam_3;Jam_tem];
        end
    end
end
Jam_4 = zeros(nBatch,size(Jam_all_sigal,2));
for i = 1:4
    Jam_4 = Jam_4 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
end
Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4];
%制作标签
class_num = 4;
zhuhe_num = 4;
label = zeros(size(Jam_All,1),class_num);
zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
for i = 1:size(zuhe_one_hot,1)
    label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
end
Jam_all = [real(Jam_All),imag(Jam_All)];
data = Jam_all;
%% 保存
% 第三份开放集组合使用显式类别列映射生成真值旁车。
legacy_class_codes_3 = {'FMB','AMA','MFT','ISRJ'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
mainFilePath = 'Jam5_3.mat';
labelFilePath = 'label5_3.mat';
truthMetadata = struct('legacy_entrypoint', 'Gen_openjamset.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition, ...
    'open_set_partition_index', 3);
truthBundle = build_legacy_truth_bundle(label, legacy_class_codes_3, ...
    atomic_truth, source_sample_index, truthMetadata);
save(mainFilePath, 'data', '-v7.3');
save(labelFilePath, 'label', '-v7.3');
save_legacy_truth_sidecar(mainFilePath, truthBundle, true);

%% 未知选用4，1345混合
Jam_all_sigal = [Jam_FMB;Jam_AMA;Jam_SawFM;Jam_ISRJ];
Jam_2 = [];
for i = 1:3
    for j = i+1:4
        Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
        Jam_2 = [Jam_2;Jam_tem];
    end
end
Jam_3 = [];
for i = 1:2
    for j = i+1:3
        for ii = j+1:4
            Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
            Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
            Jam_3 = [Jam_3;Jam_tem];
        end
    end
end
Jam_4 = zeros(nBatch,size(Jam_all_sigal,2));
for i = 1:4
    Jam_4 = Jam_4 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
end
Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4];
%制作标签
class_num = 4;
zhuhe_num = 4;
label = zeros(size(Jam_All,1),class_num);
zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
for i = 1:size(zuhe_one_hot,1)
    label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
end
Jam_all = [real(Jam_All),imag(Jam_All)];
data = Jam_all;
%% 保存
% 第四份开放集组合使用显式类别列映射生成真值旁车。
legacy_class_codes_4 = {'FMB','AMA','SawFM','ISRJ'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
mainFilePath = 'Jam5_4.mat';
labelFilePath = 'label5_4.mat';
truthMetadata = struct('legacy_entrypoint', 'Gen_openjamset.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition, ...
    'open_set_partition_index', 4);
truthBundle = build_legacy_truth_bundle(label, legacy_class_codes_4, ...
    atomic_truth, source_sample_index, truthMetadata);
save(mainFilePath, 'data', '-v7.3');
save(labelFilePath, 'label', '-v7.3');
save_legacy_truth_sidecar(mainFilePath, truthBundle, true);
%% 未知选用5，1234混合
Jam_all_sigal = [Jam_FMB;Jam_AMA;Jam_SawFM;Jam_MFT];
Jam_2 = [];
for i = 1:3
    for j = i+1:4
        Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
        Jam_2 = [Jam_2;Jam_tem];
    end
end
Jam_3 = [];
for i = 1:2
    for j = i+1:3
        for ii = j+1:4
            Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
            Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
            Jam_3 = [Jam_3;Jam_tem];
        end
    end
end
Jam_4 = zeros(nBatch,size(Jam_all_sigal,2));
for i = 1:4
    Jam_4 = Jam_4 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
end
Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4];
%制作标签
class_num = 4;
zhuhe_num = 4;
label = zeros(size(Jam_All,1),class_num);
zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
for i = 1:size(zuhe_one_hot,1)
    label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
end
Jam_all = [real(Jam_All),imag(Jam_All)];
data = Jam_all;
%% 保存
% 第五份开放集组合使用显式类别列映射生成真值旁车。
legacy_class_codes_5 = {'FMB','AMA','SawFM','MFT'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
mainFilePath = 'Jam5_5.mat';
labelFilePath = 'label5_5.mat';
truthMetadata = struct('legacy_entrypoint', 'Gen_openjamset.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition, ...
    'open_set_partition_index', 5);
truthBundle = build_legacy_truth_bundle(label, legacy_class_codes_5, ...
    atomic_truth, source_sample_index, truthMetadata);
save(mainFilePath, 'data', '-v7.3');
save(labelFilePath, 'label', '-v7.3');
save_legacy_truth_sidecar(mainFilePath, truthBundle, true);
% 脚本变量位于 base workspace；显式清理以立即恢复运行前 MATLAB path。
clear pathCleanup;

% %% 全部已知
% 
% Jam_all_sigal = [Jam_FMB;Jam_AMA;Jam_SawFM;Jam_MFT;Jam_ISRJ];
% Jam_2 = [];
% for i = 1:4
%     for j = i+1:5
%         Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
%         Jam_2 = [Jam_2;Jam_tem];
%     end
% end
% Jam_3 = [];
% for i = 1:3
%     for j = i+1:4
%         for ii = j+1:5
%             Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%             Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
%             Jam_3 = [Jam_3;Jam_tem];
%         end
%     end
% end
% 
% Jam_4 = [];
% for i = 1:2
%     for j = i+1:3
%         for ii = j+1:4
%             for jj = ii+1:5
%                 Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%                 Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:)+Jam_all_sigal((jj-1)*nBatch+1:jj*nBatch,:);
%                 Jam_4 = [Jam_4;Jam_tem];
%             end
%         end
%     end
% end
% Jam_5 = zeros(nBatch,size(Jam_all_sigal,2));
% for i = 1:5
%     Jam_5 = Jam_5 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
% end
% Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4;Jam_5];
% %制作标签
% class_num = 5;
% zhuhe_num = 5;
% label = zeros(size(Jam_All,1),class_num);
% zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
% for i = 1:size(zuhe_one_hot,1)
%     label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
% end
% Jam_all = [real(Jam_All),imag(Jam_All)];
% data = Jam_all;
% %% 保存
% save('Jam5.mat','data','-v7.3');
% save('label5.mat','label','-v7.3');









