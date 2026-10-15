clc,clear,close all;
%% 信号参数
rng('shuffle'); 
C = 3e8;
Tp = 5e-6;     %脉宽
Tpp = 20e-6;    %脉冲重复周期
duty_cycle = Tp/Tpp; %占空比
% Tpp = 100e-6;    %脉冲重复周期
fs = 100e6;      %采样率
Bw = 50e6;      %带宽
Np = Tp * fs;     %脉冲内点数
nPP = Tpp * fs;   %脉冲重复周期内点数
K = Bw / Tp;      %调频斜率
tp = (-Np/2 : Np/2-1)' / fs;
t = (-nPP/2: nPP/2-1)/fs;
%% 本振信号
sigLFM = exp(1i*pi*K*tp.^2);  %线性调频信号
SigLFM = abs(fftshift(fft(sigLFM)));
% sig_LOS = [zeros(floor(nPP-Np-900), 1);sigLFM ; zeros(floor(nPP-Np-600), 1)];  %单脉冲信号
sig_LOS = [sigLFM;zeros(floor(nPP-Np), 1)];
%% 拖引干扰
freq0=0;
 R_min=2000;
 R_max=2000;
 A_min=1;
 A_max=1;
 v_min=100;
 v_max=100;
 nBatch=1;
 nPulse=20;
 for i=1:nPulse
% out_sig(i,:) = sudujmb1(length(sig_LOS),fs,Tp,K,(10e-6)-(10e-8)*2*i, (10e-6)-(10e-8)*2*i, 1, 1,100+10*i,100+10*i,1 ,1 );
% out_sig(i,:) = sudujmb1(length(sig_LOS),fs,Tp,K,(10e-6)-(10e-8)*2, (10e-6)-(10e-8)*2, 1, 1,10000*(i-1),10000*(i-1),1 ,1 );
out_sig(i,:) = sudujmb1(length(sig_LOS),fs,Tp,K,0, 0, 1, 1,1e5*i,1e5*i,1 ,1 );
% figure,plot(real(out_sig(i,:)));
 end
%% 参数提取
canshu = Traction_duration(sig_LOS,out_sig, nPulse, fs,K);