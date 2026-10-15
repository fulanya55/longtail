% function sig_jam_out = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_median, JNR_disturbance, ...
%     jam_tau_median, jam_tau_disturbance, jam_replica_median, jam_replica_disturbance, nBatch, nRepetition, flag_addTar)
function out_sig = Data_gen_ISRJ(sig_LOS, nPulse, fs, JNR_min, JNR_max, ...
    jam_tau_min, jam_tau_max, jam_replica_min, jam_replica_max,ISRJ_conv, nBatch, nRepetition, flag_addTar , jam_delay_ISRJ_min, jam_delay_ISRJ_max)
% C3类（干扰+噪声）数据生成函数
% ----Input----
%   sig_LOS：列向量。本振信号（发射信号）。
%   nPulse:一个样本里的脉冲重复周期个数。
%   fs：标量。采样率。
%   JNR_min：标量。干噪比变化范围最小值。
%   JNR_max：标量。干噪比变化范围最大值。
%   jam_tau_min：标量。干扰采样时长最小值。
%   jam_tau_max：标量。干扰采样时长最大值。
%   jam_replica_min：标量。干扰转发次数最小值。
%   jam_replica_max：标量。干扰转发次数最大值。
%   nBatch：正整数。生成的信号批数。
%   nRepetition: 正整数，一个样本内本干扰的个数。大于等于1。
%   flag_addTar：添加目标flag
% ----Output----
%   sig_jam_out：输出数据。维数【nBatch×nSample】
% 2018.05.29 by TP
A_noise = 1/sqrt(2);     %保证噪声功率为1
sig_LOS = sig_LOS.';
nPP = length(sig_LOS);
nSample = nPP;
% sig_LOS = sig_LOS + A_noise*(randn(nSample, 1)+1i*randn(nSample, 1));
out_sig = zeros(nBatch*nPulse, nSample);
for iBatch = 1:nBatch*nPulse
    for iRep = 1:nRepetition
        %         JNR = JNR_median + JNR_median * JNR_disturbance *( rand() * 2 - 1);      %干噪比随机均匀分布
        %         A_jam = 10^(JNR/20);                   %干扰信号幅度
        %
        %         jam_tau = jam_tau_median + jam_tau_median * jam_tau_disturbance *( rand() * 2 - 1);                   %干扰采样时长均匀随机分布
        %         nReplica = randi(jam_replica_disturbance - jam_replica_median + 1) + jam_replica_median-1;  %每个采样段的重复次数,随机分布
        JNR = JNR_min + (JNR_max - JNR_min)*rand();      %干噪比随机均匀分布
        A_jam = 10^(JNR/20);                   %干扰信号幅度
        
        jam_tau = jam_tau_min + (jam_tau_max - jam_tau_min)*rand();                   %干扰采样时长均匀随机分布
        nReplica = randi(jam_replica_max - jam_replica_min + 1) + jam_replica_min-1;  %每个采样段的重复次数,随机分布
        
        jam_sig = ISRJ(repmat(sig_LOS, 1, 1), jam_tau, fs, nReplica).';                   %干扰
        out_sig(iBatch, :) = out_sig(iBatch, :) +  A_jam*jam_sig;
        
        if ISRJ_conv == 1
            sig_noise = A_noise * (randn(nPP, 1)+1i*randn(nPP, 1));
            ft = sig_noise;
            sig_jam_conv = conv(ft,out_sig(iBatch, :));
            out_sig(iBatch, :) = sig_jam_conv(1:nPP);
        end
        
    end
    jam_delay = round((jam_delay_ISRJ_min + (jam_delay_ISRJ_max - jam_delay_ISRJ_min)*rand()) * fs /1e6);
    out_sig(iBatch, :) = circshift(out_sig(iBatch, :), jam_delay);
    out_sig(iBatch, :) =  out_sig(iBatch, :) + A_noise*(randn(nSample, 1)+1i*randn(nSample, 1)).';      % 加噪声
    if flag_addTar == 1
        out_sig(iBatch, :) = Add1tar(out_sig(iBatch, :), sig_LOS, nPulse, fs);    %添加目标信号
    end
end

end

