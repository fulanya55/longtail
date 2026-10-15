clc,clear,close all;
%% 信号参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('shuffle'); 
C = 3e8;
Tp = 20e-6;     %脉宽
Tpp = 100e-6;    %脉冲重复周期
duty_cycle = Tp/Tpp; %占空比
% Tpp = 100e-6;    %脉冲重复周期
fs = 60e6;      %采样率
Bw = 40e6;      %带宽
Np = Tp * fs;     %脉冲内点数
nPP = Tpp * fs;   %脉冲重复周期内点数
K = Bw / Tp;      %调频斜率
tp = (-Np/2 : Np/2-1)' / fs;
t = (-nPP/2: nPP/2-1)'/fs;
%% 本振信号%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigLFM = 1.*exp(1i*pi*K*tp.^2);  %线性调频信号
SigLFM = abs(fftshift(fft(sigLFM)));
SigLOS = [SigLFM;zeros(nPP-Np,1)];
% sig_LOS = [zeros(floor(nPP-Np-900), 1);sigLFM ; zeros(floor(nPP-Np-600), 1)];  %单脉冲信号
% sig_LOS=awgn(sig_LOS,20,'measured');
% Sig_sinFM = abs(fftshift(fft(sig_LOS)));%频域
% data_range_sinFM = ifftshift(range_process_mf(sig_LOS,sig_LOS.',nPP));%距离域
% data_range_sinFM_dB = 10*log10(data_range_sinFM/max(data_range_sinFM));%归一化取对数距离域
% [S ,F, T] = spectrogram(sig_LOS,16,15,2048,fs,'centered');%时频域
% % 
% % figure,plot(t,real(sig_LOS)),title('阻塞干扰时域（实部）','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('幅值','FontSize',14);
% % figure,plot(-fs/2:fs/nPP:fs/2-fs/nPP,db(Sig_sinFM)),title('阻塞干扰频域','FontSize',16),xlabel('频率(Hz)','FontSize',14),ylabel('幅值(dB)','FontSize',14);
% % figure,plot(db(data_range_sinFM)),title('阻塞干扰距离域','FontSize',16),xlabel('采样点数','FontSize',14),ylabel('幅值(dB)','FontSize',14);
% % figure,imagesc(T,F,abs(S));colorbar;set(gca,'YDir','normal'),title('阻塞干扰时频域','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('频率(Hz)','FontSize',14);
% %% 信干比与干扰幅度计算%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
A_noise = 1/sqrt(2);            %噪声信号幅度，固定，不用调。噪声功率为1。
% %干扰相关参数。JNR、采样时长和干扰采样重复次数
% freq_jam = 200;
% fsin = 1e5;
% sjr=-10;%信干比
% fudu=A_noise/(10^(0.1*sjr));
% %% 密集假干扰%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nPulse = 1;         %一个样本里的脉冲重复周期个数
% n_fake_tar_min = 1; %标量，假目标数量最小值
% n_fake_tar_max = 3;
% delay_min = 5e-6; %标量，单位s，假目标时延最小值
% delay_max = 30e-6;
% amp_min = 5;       %标量，假目标幅度最小值
% amp_max = 10;
% nBatch = 2;         %正整数。生成的信号批数。
% nRepetition = 1;    %正整数，一个样本内本干扰的个数。大于等于1。
% flag_addTar = 0;    %添加目标flag。干扰里添加目标，使得场景更加逼真
% jam_delay_min = 0;
% jam_delay_max = 0;
% v_min=1e6;
% v_max=1e7;
% miji=Data_gen_Range_fake_target(sig_LOS, nPulse, fs, ...
%     delay_min, delay_max, amp_min, amp_max, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max);
% miji = miji(:,1:length(t))./(max(abs(miji')))'*fudu;
% Sig_sinFM = (abs(fftshift(fft(miji'))))';
% % for i=1
% % data_range_sinFM = ifftshift(range_process_mf(sig_LOS,miji(i,:).',nPP));
% % data_range_sinFM_dB = 10*log10(data_range_sinFM/max(data_range_sinFM));
% % [S, F, T] = spectrogram(miji(i,:),16,15,2048,fs,'centered');
% % figure,plot(t,real(miji(i,:))),title('密集假目标干扰时域（实部）','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('幅值','FontSize',14);
% % figure,plot(-fs/2:fs/nPP:fs/2-fs/nPP,db(Sig_sinFM)),title('密集假目标干扰频域','FontSize',16),xlabel('频率(Hz)','FontSize',14),ylabel('幅值(dB)','FontSize',14);
% % figure,plot(db(data_range_sinFM)),title('密集假目标干扰距离域','FontSize',16),xlabel('采样点数','FontSize',14),ylabel('幅值(dB)','FontSize',14);
% % figure,imagesc(T,F,abs(S));colorbar;set(gca,'YDir','normal'),title('密集假目标干扰时频域','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('频率(Hz)','FontSize',14);
% % 
% % end
% for i=1
% data_range_sinFM_ganrao = ifftshift(range_process_mf(sig_LOS,miji(i,:).',nPP));
% data_range_sinFM_dB = 10*log10(data_range_sinFM_ganrao/max(data_range_sinFM_ganrao));
% [S, F, T] = spectrogram(miji(i,:),16,15,2048,fs,'centered');
% figure,plot(t,real(miji(i,:)),'b'),title('密集假目标干扰时域（实部）','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('幅值','FontSize',14);
% 
% hold on
% plot(t,real(sig_LOS),'r');
% 
% figure,plot(db(data_range_sinFM_ganrao),'b'),title('密集假目标干扰距离域','FontSize',16),xlabel('采样点数','FontSize',14),ylabel('幅值(dB)','FontSize',14);
% 
% hold on
% plot(db(data_range_sinFM),'r');
% 
% figure,imagesc(T,F,abs(S));colorbar;set(gca,'YDir','normal'),title('密集假目标干扰时频域','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('频率(Hz)','FontSize',14);
% 
% hold on
% imagesc(T,F,abs(S));
% 
% end
% % %% 组合干扰目标
% % huibo=sig_LOS+miji;
fdp_min = 10;
fdp_max = 15;
amp_min = 3;
amp_max = 6;
out_sig = Data_gen_Vector_fake_target(SigLOS, 1, fs,K,Tp,...
    fdp_min, fdp_max, amp_min, amp_max, 1, 1, 0 , 0, 0);
plot(real(out_sig))


