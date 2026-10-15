function out_sig = Data_gen_Composite_SmartNoise_MultiFakeTarget(sig_LOS, nPulse, fs, JNR_min, JNR_max, ...
    n_fake_tar_min, n_fake_tar_max,delay_min_multiFakeTar, delay_max_multiFakeTar, amp_min_multiFakeTar, amp_max_multiFakeTar,jam_tau_min, jam_tau_max, jam_replica_min, jam_replica_max, nBatch,nRepetition, flag_addTar,jam_delay_min, jam_delay_max)
% 灵巧噪声干扰+密集假目标干扰的加性复合干扰
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
%   A_noise：标量。噪声信号幅度。
%   JNR_min：标量。干噪比变化范围最小值。
%   JNR_max：标量。干噪比变化范围最大值。
%   jam_tau_min：标量。干扰采样时长最小值。
%   jam_tau_max：标量。干扰采样时长最大值。
%   jam_replica_min：标量。干扰转发次数最小值。
%   jam_replica_max：标量。干扰转发次数最大值。
%   nBatch：正整数。生成的信号批数。
%   flag_addTar：添加目标flag
% ----Output----
%   out_sig： 输出数据。维数【nBatch×nSample】
% 2019.01.15 by XC

%% 参数设置
A_noise = 1/sqrt(2);
nPP = length(sig_LOS);
nSample = nPP;
jam_replica_max = ceil(jam_replica_max);
jam_replica_min =ceil(jam_replica_min);
out_sig = zeros(nBatch*nPulse, nSample);
for iBatch = 1:nBatch*nPulse
    for iRep = 1:nRepetition
    JNR = JNR_min + (JNR_max - JNR_min)*rand();      %干噪比随机均匀分布
    A_jam = 10^(JNR/20);                   %干扰信号幅度 
    jam_tau = jam_tau_min + (jam_tau_max - jam_tau_min)*rand();                   %干扰采样时长均匀随机分布
    nReplica = randi([jam_replica_min jam_replica_max]) + jam_replica_min;  %干扰采样段的重复次数,随机分布  
    tau_p = ceil(fs * jam_tau);         %转发时长对应采样点数
    nJamSegment = ceil(nSample/(1+nReplica)/tau_p);   %干扰截取的段数
    tmp_win = [ones(tau_p, 1) ; zeros(tau_p*nReplica, 1)];    %一段干扰的窗
    jam_win = repmat(tmp_win, nJamSegment ,1);  %所有干扰的窗
    jam_win = jam_win(1:nSample);     %去掉多余的点，使长度一致
    jam_sample = jam_win .* repmat(sig_LOS, 1, 1); %干扰采样
    jam_sig1 = zeros(nSample, 1);
    for ii = 1:nReplica
        phase_noise = jam_win .* randn(nSample, 1);  %附加在信号相位上的噪声
        tmp_noise = [zeros(ii*tau_p, 1) ; phase_noise(1:end-ii*tau_p)] ;
        tmp_data = [zeros(ii*tau_p, 1) ; jam_sample(1:end-ii*tau_p)] ;
        jam_sig1 = jam_sig1 + tmp_data .* exp(1i*tmp_noise);   %干扰信号
    end
    
    sig_noise = A_noise * (randn(nSample, 1)+1i*randn(nSample, 1));      %噪声信号
    jam_sig_noise1 = (A_jam*jam_sig1 + sig_noise).';
    
    
        jam_sig2 = zeros(1,nSample);
    n_fake_tar = randi(n_fake_tar_max - n_fake_tar_min + 1) + n_fake_tar_min-1;
    for ii = 1:n_fake_tar %逐个生成
        delay = delay_min_multiFakeTar + (delay_max_multiFakeTar-delay_min_multiFakeTar)*rand();
        amp = amp_min_multiFakeTar + (amp_max_multiFakeTar-amp_min_multiFakeTar)*rand();
        n_delay = round(fs*delay);
        jam_sig2 = jam_sig2 + amp * circshift(repmat(sig_LOS, 1, 1), n_delay).';           %干扰    
    end
    end
    jam_sig_noise2 = jam_sig2 +  A_noise*(randn(1, nSample) + 1i*randn(1, nSample));  
    out_sig(iBatch, :) = jam_sig_noise1 + jam_sig_noise2;
        jam_delay = round((jam_delay_min + (jam_delay_max - jam_delay_min)*rand()) * fs /1e6);
    out_sig(iBatch, :) = circshift(out_sig(iBatch, :), jam_delay);
    if flag_addTar == 1
%         jam_sig_noise1 = Add1tar(jam_sig_noise1, sig_LOS, nPulse, fs, 0);    %添加目标信号
%         jam_sig_noise2 = Add1tar(jam_sig_noise2, sig_LOS, nPulse, fs, 0);    %添加目标信号
        out_sig(iBatch, :) = Add1tar(out_sig(iBatch, :), sig_LOS, nPulse, fs);    %添加目标信号
    end
    
end

end