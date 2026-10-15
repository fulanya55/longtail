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
%% 干扰参数设置
% %参数组合1
J_s_min = 20;
J_s_max = 38.5; 
J_d_min = 20;
J_d_max = 40;
% % %参数组合2
% % J_s_min = 30;
% % J_s_max = 50;
% % J_d_min = 20;
% % J_d_max = 40;
% %参数组合3
% J_s_min = 40;
% J_s_max = 60;
% J_d_min = 20;
% J_d_max = 40;
%%%单一测试
% J_s_min = 30;
% J_s_max = 30; 
% J_d_min = 30;
% J_d_max = 30;

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

% %% 单类汇总数据
% Jam_all_sigal = [Jam_FMB;Jam_AMA;Jam_SawFM;Jam_SinFM;Jam_SN;Jam_MFT;Jam_vft;Jam_ISRJ];
% %% 复合干扰生成
% %FMB:1 AMA:2 SawFM:3 SinFM:4 SN:5 MFT:6 Vft:7 ISRJ:8
% Jam_2 = [];
% for i = 1:7
%     for j = i+1:8
%         Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:);
%         Jam_2 = [Jam_2;Jam_tem];
%     end
% end
% 
% Jam_3 = [];
% for i = 1:6
%     for j = i+1:7
%         for ii = j+1:8
%             Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%             Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:);
%             Jam_3 = [Jam_3;Jam_tem];
%         end
%     end
% end
% 
% Jam_4 = [];
% for i = 1:5
%     for j = i+1:6
%         for ii = j+1:7
%             for jj = ii+1:8
%                 Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%                 Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:)+Jam_all_sigal((jj-1)*nBatch+1:jj*nBatch,:);
%                 Jam_4 = [Jam_4;Jam_tem];
%             end
%         end
%     end
% end
% 
% Jam_5 = [];
% for i = 1:4
%     for j = i+1:5
%         for ii = j+1:6
%             for jj = ii+1:7
%                 for iii = jj+1:8
%                     Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%                     Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:)+Jam_all_sigal((jj-1)*nBatch+1:jj*nBatch,:)+Jam_all_sigal((iii-1)*nBatch+1:iii*nBatch,:);
%                     Jam_5 = [Jam_5;Jam_tem];
%                 end
%             end
%         end
%     end
% end
% 
% Jam_6 = [];
% for i = 1:3
%     for j = i+1:4
%         for ii = j+1:5
%             for jj = ii+1:6
%                 for iii = jj+1:7
%                     for jjj = iii+1:8
%                         Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%                         Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:)+Jam_all_sigal((jj-1)*nBatch+1:jj*nBatch,:)+Jam_all_sigal((iii-1)*nBatch+1:iii*nBatch,:)+...
%                         Jam_all_sigal((jjj-1)*nBatch+1:jjj*nBatch,:);
%                         Jam_6 = [Jam_6;Jam_tem];
%                     end
%                 end
%             end
%         end
%     end
% end
% 
% Jam_7 = [];
% for i = 1:2
%     for j = i+1:3
%         for ii = j+1:4
%             for jj = ii+1:5
%                 for iii = jj+1:6
%                     for jjj = iii+1:7
%                         for iiii = jjj+1:8
%                             Jam_tem(1:nBatch,:) = Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:)+ Jam_all_sigal((j-1)*nBatch+1:j*nBatch,:)+ ...
%                             Jam_all_sigal((ii-1)*nBatch+1:ii*nBatch,:)+Jam_all_sigal((jj-1)*nBatch+1:jj*nBatch,:)+Jam_all_sigal((iii-1)*nBatch+1:iii*nBatch,:)+...
%                             Jam_all_sigal((jjj-1)*nBatch+1:jjj*nBatch,:)+Jam_all_sigal((iiii-1)*nBatch+1:iiii*nBatch,:);
%                             Jam_7 = [Jam_7;Jam_tem];
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end
% Jam_8 = zeros(nBatch,size(Jam_2,2));
% for i = 1:8
%     Jam_8 = Jam_8 + Jam_all_sigal((i-1)*nBatch+1:i*nBatch,:);
% end
% Jam_All = [Jam_all_sigal;Jam_2;Jam_3;Jam_4;Jam_5;Jam_6;Jam_7;Jam_8];
% %% 制作标签
% class_num = 8;
% zhuhe_num = 8;
% label = zeros(size(Jam_All,1),class_num);
% zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
% for i = 1:size(zuhe_one_hot,1)
%     label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
% end
% Jam_all = [real(Jam_All),imag(Jam_All)];
% data = Jam_all ;
% %% 保存
% save('Jam8_zh1.mat','data','-v7.3');
% save('label8_zh1.mat','label','-v7.3');
%% 单类汇总数据
% Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_SinFM,Jam_SN,Jam_MFT,Jam_vft,Jam_ISRJ};
% Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_SinFM,Jam_MFT,Jam_vft,Jam_ISRJ};
Jam_all_sigal = {Jam_FMB,Jam_SawFM,Jam_MFT,Jam_ISRJ};
arrays = Jam_all_sigal;
Jam_all = [];
%% 复合干扰生成
%FMB:1 AMA:2 SawFM:3 MFT:4 Vft:5 ISRJ:6 zh0
%FMB:1 SawFM:2 SinFM:3 MFT:4 Vft:5 ISRJ:6 zh1
%%FMB:1 SawFM:2 SinFM:3 SN:4 MFT:5 Vft:6 ISRJ:7
for k = 1:length(arrays)
    % 获取当前组合的元素数量
    currentCombinationSize = k;
    
    % 获取所有可能的组合索引
    combinationIndices = nchoosek(1:length(arrays), currentCombinationSize);
    
    % 对于每个组合索引，提取对应的数组并合并
    for i = 1:size(combinationIndices, 1)
        selectedArrays = combinationIndices(i, :);
        combinedArray = [];
        Jam_tem = zeros(nBatch, nPP);
        for j = 1:length(selectedArrays)
%             combinedArray = [combinedArray;arrays{selectedArrays(j)}];
            combinedArray = arrays{selectedArrays(j)};
            Jam_tem = Jam_tem+combinedArray;
        end
        Jam_all = [Jam_all;Jam_tem];
%         % 输出当前组合
%         fprintf('组合大小为 %d 的第 %d 个组合：\n', currentCombinationSize, i);
%         disp(combinedArray);
%         fprintf('\n');
    end
end

%% 制作标签
class_num = 4;
zhuhe_num = 4;
label = zeros(size(Jam_all,1),class_num);
zuhe_one_hot = zuhe_label(class_num, zhuhe_num);
for i = 1:size(zuhe_one_hot,1)
    label((i-1)*nBatch+1:i*nBatch,:) = repmat(zuhe_one_hot(i,:),nBatch,1);
end
Jam_all = [real(Jam_all),imag(Jam_all)];
data = Jam_all ;
%% 保存
save('D:\干扰数据\复合干扰数据\Jam4_zh0_tttt.mat','data','-v7.3');
save('D:\干扰数据\复合干扰数据\label4_zh0_tttt.mat','label','-v7.3');






