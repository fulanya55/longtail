clc,clear,close all;
%% 信号参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('shuffle'); 
C = 3e8;
Tp = 20e-6;     %脉宽
Tpp = 20e-6;    %脉冲重复周期
duty_cycle = Tp/Tpp; %占空比
% Tpp = 100e-6;    %脉冲重复周期
fs = 100e6;      %采样率
Bw = 40e6;      %带宽
Np = Tp * fs;     %脉冲内点数
nPP = Tpp * fs;   %脉冲重复周期内点数
K = Bw / Tp;      %调频斜率
tp = (-Np/2 : Np/2-1)' / fs;
t = (-nPP/2: nPP/2-1)/fs;
%% 本振信号%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigLFM = 1.*exp(1i*pi*K*tp.^2);  %线性调频信号
SigLFM = abs(fftshift(fft(sigLFM)));
% sig_LOS = [zeros(floor(nPP-Np-900), 1);sigLFM ; zeros(floor(nPP-Np-600), 1)];  %单脉冲信号
% sig_LOS=awgn(sig_LOS,20,'measured');
%% 干扰
n_min=10;
n_max=20;
A_min=1;
A_max=5;
T_min=20e-8;
T_max=20e-6;
tao_min=20e-8;
tao_max=10e-6;
miji =  zaluan_jam(sigLFM,nPP,fs,Bw,Tpp,n_min,n_max,A_min,A_max,T_min,T_max,tao_min,tao_max);
Sig_sinFM = (abs(fftshift(fft(miji'))))';
for i=1
data_range_sinFM = ifftshift(range_process_mf(sigLFM,miji(i,:).',nPP));
data_range_sinFM_dB = 10*log10(data_range_sinFM/max(data_range_sinFM));
[S, F, T] = spectrogram(miji(i,:),16,15,2048,fs,'centered');
figure,plot(t,real(miji(i,:))),title('杂乱脉冲干扰时域（实部）','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('幅值','FontSize',14);
figure,plot(-fs/2:fs/nPP:fs/2-fs/nPP,db(Sig_sinFM)),title('杂乱脉冲干扰频域','FontSize',16),xlabel('频率(Hz)','FontSize',14),ylabel('幅值(dB)','FontSize',14);
figure,plot(db(data_range_sinFM)),title('杂乱脉冲干扰距离域','FontSize',16),xlabel('采样点数','FontSize',14),ylabel('幅值(dB)','FontSize',14);
figure,imagesc(T,F,abs(S));colorbar;set(gca,'YDir','normal'),title('杂乱脉冲干扰时频域','FontSize',16),xlabel('时间(s)','FontSize',14),ylabel('频率(Hz)','FontSize',14);

end