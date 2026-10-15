function out_sig = Data_gen_JsawFM(sig_LOS, JNR_min, JNR_max, Bw,K, ...
    F_sweep_min, F_sweep_max , jam_delay_min, jam_delay_max,fs, nRepetition, nPulse, nBatch, flag_addTar)
%%C3类（干扰+噪声）数据生成函数
% ----Input----
%   sig_LOS：列向量。本振信号（发射信号）。
%   nPulse:一个样本里的脉冲重复周期个数。
%   fs：标量。采样率。
%   K:扫频干扰的扫频斜率
%   JNR_min：标量。干噪比变化范围最小值。
%   JNR_max：标量。干噪比变化范围最大值。
%   freq_jam :扫频干扰带宽
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
        %% 干扰参数
        JNR = JNR_min + (JNR_max - JNR_min)*rand();
        F_sweep = (F_sweep_min + (F_sweep_max - F_sweep_min)*rand())*Bw;
        np_sweep = floor(F_sweep/K*fs) ; %一个干扰扫频周期内的点数
        k = K;
        n = ceil(nPP/np_sweep);
%         t_sweep = (-np_sweep/2 : np_sweep/2-1)/fs ;  %一个干扰扫频周期的时间轴
        t_sweep = (1 : np_sweep)/fs ;
        Uj = 10^(JNR/20);      %振幅
         %%扫频干扰
         f_saw = k*t_sweep;
        sig_sweep = exp(1j*pi*f_saw.*t_sweep);
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