function out_sig = Data_gen_NoiseFM_Block(sig_LOS, nPulse, JNR_min, JNR_max, ...
    L_min, L_max, sig_bandwidth, freq0, fs, nPP, nBatch, nRepetition, flag_addTar , jam_delay_min, jam_delay_max)
% 阻塞式 噪声调频干扰 生成函数
%----输入----
% sig_LOS:本振信号
%   nPulse:一个样本里的脉冲重复周期个数。
% JNR_min：          标量。干噪比变化范围最小值。
% JNR_max：          标量。干噪比变化范围最大值。
%   L_min:           整数。干扰带宽比信号带宽最小值5
%   L_max:           整数。干扰带宽比信号带宽最大值8
% sig_bandwidth:   雷达信号带宽，单位Hz
% freq0:           信号的中心频率，单位Hz
% fs：             采样频率，单位Hz
% nPP:          干扰信号的采样点数。
% nBatch：          正整数。生成的信号批数。
% nRepetition: 正整数，一个样本内本干扰的个数。大于等于1。
% flag_addTar：添加目标flag
%----输出----
% out_sig：       single类型输出数据。维数【nBatch×nSample】
%% 参数设置
nSample = nPP;
A_noise = 1/sqrt(2);
phi = 0;               %相位
t = (0:nSample-1)./fs;
% freq0 = 20e6 * rand() - 10e6;
% L_min = 5;  %干扰带宽比信号带宽。阻塞式干扰该值应该大于5
% L_max = 8;
mfe = 10;             %有效调频指数
out_sig = zeros(nBatch*nPulse, nSample);
for iBatch = 1:nBatch*nPulse
    for iRep = 1:nRepetition
        %% 噪声带宽计算
        L = L_min + (L_max - L_min)*rand();
        jam_sig_BW = L*sig_bandwidth;            %期望的干扰信号的干扰带宽，半功率干扰带宽
        fde = jam_sig_BW/(2*sqrt(2*log(2)));%有效调频带宽
        noise_BW = fde/mfe;    %调制噪声的带宽。
        %% 幅度确定
        JNR = JNR_min + (JNR_max - JNR_min)*rand();      %干噪比随机均匀分布
        Uj = 10^(JNR/20);      %振幅
        %% 色噪声生成
        % 高斯白噪声
        Gauss_noise =randn(1, nSample) ; %产生均值为0、方差为1的高斯白噪声
        G_noise_fft = fft(Gauss_noise);     %FFT
        % 通过滤波器滤波（直接频域截取），得到色噪声
        nFilter = round( noise_BW/fs/2*nSample);
        f_filter = zeros(1, nSample);
        f_filter(1:nFilter) = 1;
        f_filter(end-nFilter+1 :end) = 0;
        color_noise_fft = G_noise_fft.*f_filter;            %对白噪声进行加窗处理，得到频域上的高斯色噪声
        color_noise = real(ifft(color_noise_fft));                %得到高斯色噪声序列，即为调制噪声
        %% 干扰生成
        sigma_n = std(color_noise);             %计算色噪声序列的标准差 （sigma_n）^2为调制噪声的功率。
        KFM = fde/sigma_n;                      %调频斜率
        freq_modulation = cumsum(color_noise/fs);
        jam_sig = Uj * exp(1j*(2*pi*freq0*t + 2*pi*KFM*freq_modulation + phi));  %在时域构造噪声调频信号
        out_sig(iBatch, :) = out_sig(iBatch, :) + jam_sig;
    end
        jam_delay = round((jam_delay_min + (jam_delay_max - jam_delay_min)*rand()) * fs );
    out_sig(iBatch, :) = circshift(out_sig(iBatch, :), jam_delay);
    out_sig(iBatch, :) =  out_sig(iBatch, :) + A_noise*(randn(1, nSample) + 1i*randn(1, nSample));  %加噪声
    if flag_addTar == 1
        out_sig(iBatch, :) = Add1tar(out_sig(iBatch, :), sig_LOS, nPulse, fs);    %添加目标信号
    end
end
end

