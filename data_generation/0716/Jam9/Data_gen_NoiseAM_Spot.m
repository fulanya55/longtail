function out_sig = Data_gen_NoiseAM_Spot(sig_LOS, nPulse, JNR_min, JNR_max, ...
    L_min, L_max, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max)
% 瞄准式 噪声调幅干扰 生成函数
%----输入----
%   sig_LOS:本振信号
%   nPulse:            一个样本里的脉冲重复周期个数。
%   JNR_min：          标量。干噪比变化范围最小值。
%   JNR_max：          标量。干噪比变化范围最大值。
%   L_min:             整数。干扰带宽比信号带宽最小值2
%   L_max:             整数。干扰带宽比信号带宽最大值4
%   sig_bandwidth:     雷达信号带宽，单位Hz
%   freq0:             信号的中心频率，单位Hz。通常设置为0。零中频信号。
%   fs：               采样频率，单位Hz
%   nPoint:            干扰信号的采样点数。
%   nBatch：           正整数。生成的信号批数。
%   nRepetition:       正整数，一个样本内本干扰的个数。大于等于1。
%   flag_addTar：      添加目标flag
%----输出----
%   out_sig：          输出数据。维数【nBatch×nSample】
%% 参数设置
nSample = nPP;
A_noise = 1/sqrt(2);
U0 = 3;         %振幅，固定
phi = 0;        %相位
% freq0 = 20e6 * rand() - 10e6;
% L_min = 2;    %干扰带宽比信号带宽
% L_max = 4;
t = (0:nSample-1)/fs;
out_sig = zeros(nBatch*nPulse, nSample);
for iBatch = 1:nBatch*nPulse
    for iRep = 1:nRepetition
        %% 干扰带宽确定
        L = L_min + (L_max - L_min)*rand();     %干扰的带宽倍数。瞄准式干扰该值为2~5
        jam_sig_BW = L * sig_bandwidth;           %调制噪声带宽。对噪声调幅干扰来说，干扰信号带宽=调制噪声带宽
        %% 干噪比随机均匀分布
        JNR = JNR_min + (JNR_max - JNR_min)*rand();
        %% 色噪声生成
        % 高斯白噪声
        Gauss_noise = randn(1, nSample) ;    %实数，高斯白噪声
        G_noise_fft = fft(Gauss_noise);
        % 通过滤波器滤波（直接频域截取），得到色噪声
        nFilter = round(jam_sig_BW/fs/2*nSample);   %频域滤波点数
        f_filter = zeros(1, nSample);
        if nFilter>=nSample
            f_filter(1:nSample) = 1;
        else
            f_filter(1:nFilter) = 1;
            f_filter(end-nFilter+1 :end) = 0;
        end
        color_noise_fft = G_noise_fft.*f_filter;    %对白噪声进行加窗处理，得到频域上的高斯色噪声
        color_noise = ifft(color_noise_fft);        %得到高斯色噪声序列，即为调制噪声
        color_noise = real( sqrt(fs/jam_sig_BW) * color_noise);    %调整一下幅度
        
        %% 干扰生成
        jam_amp = 10.^(JNR/20)/sqrt(U0^2+1) *(U0+color_noise) ;
        jam_sig = jam_amp.* exp(1j*(2*pi*freq0*t + phi));  %在时域构造噪声调幅信号
        out_sig(iBatch, :) = out_sig(iBatch, :) + jam_sig;
    end
    jam_delay = round((jam_delay_min + (jam_delay_max - jam_delay_min)*rand()) * fs );
    out_sig(iBatch, :) = circshift(out_sig(iBatch, :), jam_delay);
    out_sig(iBatch, :) = out_sig(iBatch, :) + A_noise*(randn(1, nSample) + 1i*randn(1, nSample)); %加噪声
    if flag_addTar == 1
        out_sig(iBatch, :) = Add1tar( out_sig(iBatch, :), sig_LOS, nPulse, fs);    %添加目标信号
    end
end
end


