function out_sig = Data_gen_Multi_fake_target(sig_LOS, nPulse, fs, n_fake_tar_min, n_fake_tar_max, ...
    delay_min, delay_max, amp_min, amp_max, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max)
% 密集假目标干扰生成函数
% ----Input----
%   sig_LOS：列向量。本振信号（发射信号）。
%   nPulse:一个样本里的脉冲重复周期个数。
%   fs：标量。采样率。
%   n_fake_tar_min:标量，假目标数量最小值
%   n_fake_tar_max:标量，假目标数量最大值
%   delay_min：标量，单位s，假目标时延最小值
%   delay_max：标量，单位s，假目标时延最大值
%   amp_min：标量，假目标幅度最小值
%   amp_max：标量，假目标幅度最大值
%   nBatch： 正整数。生成的信号批数。
%   nRepetition: 正整数，一个样本内本干扰的个数。大于等于1。
%   flag_addTar：添加目标flag
% ----Output----
%   out_sig： 输出数据。维数【nBatch×nSample】
sig_LOS = sig_LOS.';
nPP = length(sig_LOS); %信号长度
A_noise = 1/sqrt(2); %噪声功率
nSample =nPP;
% sig_LOS = sig_LOS + A_noise*(randn(nSample, 1)+1i*randn(nSample, 1));
out_sig = zeros(nBatch*nPulse, nSample);
for iBatch = 1:nBatch*nPulse
    for iRep = 1:nRepetition
        jam_sig = zeros(1,nSample);
        n_fake_tar = (n_fake_tar_max - n_fake_tar_min + 1) + n_fake_tar_min;
        for ii = 1:n_fake_tar %逐个生成
            delay = delay_min + (delay_max-delay_min)*rand();
            amp = amp_min + (amp_max-amp_min)*rand();
            n_delay = round(fs*delay);
            jam_sig = jam_sig + amp * circshift(repmat(sig_LOS, 1, 1), n_delay);           %干扰
        end
        
        out_sig(iBatch, :) = out_sig(iBatch, :) + jam_sig;
    end
    jam_delay = round((jam_delay_min + (jam_delay_max - jam_delay_min)*rand()) * fs /1e6);
    out_sig(iBatch, :) = circshift(out_sig(iBatch, :), jam_delay);
    out_sig(iBatch, :) = out_sig(iBatch, :) +  A_noise*(randn(1, nSample) + 1i*randn(1, nSample));
    if flag_addTar == 1
        out_sig(iBatch, :) = Add1tar(out_sig(iBatch, :), sig_LOS, nPulse, fs);    %添加目标信号
    end
end
end
