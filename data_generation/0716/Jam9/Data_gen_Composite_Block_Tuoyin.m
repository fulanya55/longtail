function out_sig = Data_gen_Composite_Block_Tuoyin(sig_LOS, nPulse, fs,freq0, JNR_min, JNR_max, ...
    L_min,L_max, B,Tp,Tr,A_min,A_max,R_min,R_max,v_min,v_max, nBatch,nRepetition ,jam_delay_min, jam_delay_max)
% 阻塞式 噪声调频 加 托引干扰 生成函数
%----输入----
% sig_LOS:本振信号
%   nPulse:一个样本里的脉冲重复周期个数。
% JNR_min：          标量。干噪比变化范围最小值。
% JNR_max：          标量。干噪比变化范围最大值。
% L_min:           整数。干扰带宽比信号带宽最小值5
% L_max:           整数。干扰带宽比信号带宽最大值8
% sig_bandwidth:   雷达信号带宽，单位Hz
% freq0:           信号的中心频率，单位Hz
% fs：             采样频率，单位Hz
% nPP:          干扰信号的采样点数。
% nBatch：          正整数。生成的信号批数。
% nRepetition: 正整数，一个样本内本干扰的个数。大于等于1。
% flag_addTar：添加目标flag


%----输出----
% out_sig：       single类型输出数据。维数【4nBatch×nSample】
%%参数设置
c = 3e8;    %光速
ts=1/fs;
del_t=0;%转发时延
f0 =freq0;
K=B/Tp;%调频斜率
Np = Tp*fs; 
npp = Tr*fs;
nSample = length(sig_LOS);
A_noise = 1/sqrt(2);
phi = 0;               %相位
t = (0:nSample-1)./fs;
mfe = 10;             %有效调频指数
%%参考信号
tp = (0:Np-1)/fs;
sigLFM = exp(1j*2*pi*f0*tp+1j*pi*K*tp.^2);%线性调频信号
sigLFM = [sigLFM,zeros(1,nSample-Np)];
out_sig_Block = zeros(nBatch*nPulse/4,nSample);
out_sig_tuoyin = zeros(nPulse*nBatch,nSample);
out_sig = zeros(nPulse*nBatch,nSample);
%%干扰生成
for iBatch = 1:nBatch*nPulse/4
    for iRep = 1:nRepetition
        %% 噪声带宽计算
        L = L_min + (L_max - L_min)*rand();
        jam_sig_BW = L*B;            %期望的干扰信号的干扰带宽，半功率干扰带宽
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
        out_sig_Block(iBatch, :) = out_sig_Block(iBatch, :) + jam_sig;
    end
    jam_delay = round((jam_delay_min + (jam_delay_max - jam_delay_min)*rand()) * fs);
    out_sig_Block(iBatch, :) = circshift(out_sig_Block(iBatch, :), jam_delay);
    %%拖引干扰生成
    R = R_min + (R_max - R_min)*rand();
    A = A_min + (A_max - A_min)*rand();
    v = v_min + (v_max - v_min)*rand();
    %%***********停拖期（0-1秒）*****************************************************
    t1=1;
    Signal_Jam_0 = A.*sigLFM;
    out_sig_tuoyin(4*iBatch-3,:) = Signal_Jam_0;
    %%*************拖引期（1-2秒时）****************
    tt=2;
    RJ=R+v*(tt-t1);
    dlt_t=2*v*(tt-t1)/c;
    delayNum=fix(dlt_t*fs);
    tp = (0-delayNum:Np-delayNum-1)/fs;
    Signal_Jam_1 = A.*exp(1j*pi*K*tp.^2);
    Signal_Jam_1 = [Signal_Jam_1,zeros(1,nSample-Np)];
    out_sig_tuoyin(4*iBatch-2,:) = Signal_Jam_1;
    %%**********************拖引期（1-5秒时）******************************************
    tt=5;
    RJ=R+v*(tt-t1);
    dlt_t=2*v*(tt-t1)/c;
    delayNum=fix(dlt_t*fs);
    tp = (0-delayNum:Np-delayNum-1)/fs;
    Signal_Jam_2 = A.*exp(1j*pi*K*tp.^2);
    Signal_Jam_2 = [Signal_Jam_2,zeros(1,nSample-Np)];
    out_sig_tuoyin(4*iBatch-1,:) = Signal_Jam_2;
    %***********************拖引期（1-10秒时）*****************************************
    tt=10;
    RJ=R+v*(tt-t1);
    dlt_t=2*v*(tt-t1)/c;
    delayNum=fix(dlt_t*fs);
    Signal_Jam_3 = A.*exp(1j*pi*K*tp.^2);
    Signal_Jam_3 = [Signal_Jam_3,zeros(1,nSample-Np)];
    out_sig_tuoyin(4*iBatch,:) = Signal_Jam_3;
    %%组合干扰
    for i = 1:4
    out_sig(4*iBatch-i+1,:) = out_sig_tuoyin(4*iBatch-i+1,:)+out_sig_Block(iBatch, :);
    out_sig(4*iBatch-i+1,:) = out_sig(4*iBatch-i+1,:)+ A_noise*(randn(1, nSample) + 1i*randn(1, nSample));  %加噪声
    end
end

end