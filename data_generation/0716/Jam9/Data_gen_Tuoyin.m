function [out_sig,out_sig_pc] = Data_gen_Tuoyin(B,Tp,freq0,fs,Tr,...
                   R_min, R_max, A_min,A_max, v_min, v_max, nBatch,nPulse)
%%C7类（干扰+噪声）数据生成函数
% ----Input----
%   sig_LOS：列向量。本振信号（发射信号）。
%   nPulse:一个样本里的脉冲重复周期个数。
%   B:雷达信号带宽
%   Tr:脉冲重复周期
%   freq0:信号中心频率
%   fs：标量。采样率。
%   Tp:雷达信号脉冲宽度
%   R_min：标量。弹目距离范围最小值。
%   R_max：标量。弹目距离范围最大值。
%   A_min: 标量。拖引干扰幅度范围最小值。
%   A_max: 标量。拖引干扰幅度范围最大值。
%   v_min: 标量。拖引干扰速度范围最小值。
%   v_max: 标量。拖引干扰速度范围最大值。
%   nBatch：正整数。生成的信号批数。
% ----Output----
%   sig_jam_out：输出数据。维数【4nBatch×nSample】
%%参数设置
c = 3e8;    %光速
ts=1/fs;
del_t=0;%转发时延
f0 =freq0;
K=B/Tp;%调频斜率
Np = Tp*fs; 
npp = Tr*fs;
nSample = npp;
%%参考信号
tp = (-Np/2:Np/2-1)/fs;
sigLFM = exp(1j*2*pi*f0*tp+1j*pi*K*tp.^2);%线性调频信号
sigLFM = [sigLFM,zeros(1,nSample-Np)];
out_sig = zeros(nBatch*nPulse,nSample);
for iBatch = 1:nBatch*nPulse/4
    R = R_min + (R_max - R_min)*rand();
    A = A_min + (A_max - A_min)*rand();
    v = v_min + (v_max - v_min)*rand();
    %%***********停拖期（0-1秒）*****************************************************
    t1=1;
    Signal_Jam_0 = A.*sigLFM;
    Signal_Jam_0_pc = ifftshift(range_process_mf(sigLFM,Signal_Jam_0,npp));
    out_sig(4*iBatch-3,:) = Signal_Jam_0;
    out_sig_pc(4*iBatch-3,:) = Signal_Jam_0_pc; 
    %%*************拖引期（1-2秒时）****************
    tt=2;
    RJ=R+v*(tt-t1);
    dlt_t=2*v*(tt-t1)/c;
    delayNum=fix(dlt_t*fs);
    tp = (-Np/2-delayNum:Np/2-delayNum-1)/fs;
    Signal_Jam_1 = A.*exp(1j*pi*K*tp.^2);
    Signal_Jam_1 = [Signal_Jam_1,zeros(1,nSample-Np)];
    Signal_Jam_1_pc = ifftshift(range_process_mf(sigLFM,Signal_Jam_1,npp));
    out_sig(4*iBatch-2,:) = Signal_Jam_1;
    out_sig_pc(4*iBatch-2,:) = Signal_Jam_1_pc; 
    %%**********************拖引期（1-5秒时）******************************************
    tt=5;
    RJ=R+v*(tt-t1);
    dlt_t=2*v*(tt-t1)/c;
    delayNum=fix(dlt_t*fs);
    tp = (-Np/2-delayNum:Np/2-delayNum-1)/fs;
    Signal_Jam_2 = A.*exp(1j*pi*K*tp.^2);
    Signal_Jam_2 = [Signal_Jam_2,zeros(1,nSample-Np)];
    Signal_Jam_2_pc = ifftshift(range_process_mf(sigLFM,Signal_Jam_2,npp));
    out_sig(4*iBatch-1,:) = Signal_Jam_2;
    out_sig_pc(4*iBatch-1,:) = Signal_Jam_2_pc;
    %***********************拖引期（1-10秒时）*****************************************
    tt=10;
    RJ=R+v*(tt-t1);
    dlt_t=2*v*(tt-t1)/c;
    delayNum=fix(dlt_t*fs);
    tp = (-Np/2-delayNum:Np/2-delayNum-1)/fs;
    Signal_Jam_3 = A.*exp(1j*pi*K*tp.^2);
    Signal_Jam_3 = [Signal_Jam_3,zeros(1,nSample-Np)];
    Signal_Jam_3_pc = ifftshift(range_process_mf(sigLFM,Signal_Jam_3,npp));
    out_sig(4*iBatch,:) = Signal_Jam_3;
    out_sig_pc(4*iBatch,:) = Signal_Jam_3_pc;
end
end