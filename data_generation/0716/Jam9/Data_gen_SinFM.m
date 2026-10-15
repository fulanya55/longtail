function out_sig = Data_gen_SinFM(sig_LOS, JNR_min, JNR_max, Bw, freq_jam, fsin, ...
    F_sweep_min, F_sweep_max , jam_delay_min, jam_delay_max, fs, nRepetition, nPulse, nBatch, flag_addTar)
%%C3类（干扰+噪声）数据生成函数
% ----Input----
%   sig_LOS：列向量。本振信号（发射信号）。
%   nPulse:一个样本里的脉冲重复周期个数。
%   fs：标量。采样率。
%   JNR_min：标量。干噪比变化范围最小值。
%   JNR_max：标量。干噪比变化范围最大值。
%   freq_jam :扫频干扰中心频点
%   fsin:正弦波频率
%   F_sweep_min: 扫频干扰频率变化范围最小倍数。
%   F_sweep_max: 扫频干扰频率变化范围最大倍数。
%   nBatch：正整数。生成的信号批数。
%   nRepetition: 正整数，一个样本内本干扰的个数。大于等于1。
%   flag_addTar：添加目标flag
% ----Output----
%   sig_jam_out：输出数据。维数【nBatch×nSample】
%%参数设置
nPP = length(sig_LOS);  %生成的数据总长度
A_noise = 1/sqrt(2);
nSample = nPP;
out_sig = zeros(nBatch*nPulse,nSample);
for iBatch = 1:nBatch*nPulse
    for iRep = 1:nRepetition
        JNR = JNR_min + (JNR_max - JNR_min)*rand();
        F_sweep = (F_sweep_min + (F_sweep_max - F_sweep_min)*rand())*Bw;
%         k = freq_jam*F_sweep;
        np_sweep = floor(F_sweep/fs*nPP) ; %一个干扰扫频周期内的点数
        n = ceil(nPP/np_sweep);%干扰段数
%         t_sweep = (-np_sweep/2 : np_sweep/2-1)/fs ;  %一个干扰扫频周期的时间轴
        t_sweep = (1 : np_sweep)/fs ;
        Uj = 10^(JNR/20);      %振幅
        %sig_sweep = exp(1j*k/fsin*(cos(2*pi*fsin*t_sweep) - cos(2*pi*fsin*t_sweep(1))));
        f = freq_jam + F_sweep*cos(2*pi*fsin*t_sweep);
        sig_sweep = exp(1j*2*pi*f.*t_sweep);
        
        sig_sweep = repmat(sig_sweep,[1,n]);
        sig_sweep = sig_sweep(1:nSample);
        out_sig(iBatch,:) = out_sig(iBatch,:)+Uj*sig_sweep;
    end
    jam_delay = round((jam_delay_min + (jam_delay_max - jam_delay_min)*rand()) * fs );
    out_sig(iBatch, :) = circshift(out_sig(iBatch, :), jam_delay);
    out_sig(iBatch, :) =  out_sig(iBatch, :) + A_noise*(randn(1,nSample)+1i*randn(1,nSample));      % 加噪声
if flag_addTar == 1
        out_sig(iBatch, :) = Add1tar(out_sig(iBatch, :), sig_LOS, nPulse, fs);    %添加目标信号
end

end

end