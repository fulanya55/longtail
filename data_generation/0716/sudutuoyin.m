function out_sig = sudutuoyin(Bw,fs,Tw,Np,t_delay1,t_delay2,amp_min,amp_max,fd)

%SUDUTUOYIN 速度拖引
% Tw = 20e-3;  %脉宽    无模糊range3000Km
% fs = 100e3;  %采样率
% amp_min = 4.5;
% amp_max = 5; 


% Np = 512;   %一个积累周期内的脉冲数
% Bw = 20e3;   %带宽,range分辨率7.5Km

% t_delay1 = 1e-3;    %时延 对应range 150Km 20个range门
% t_delay2 = 2e-3;    %时延 对应range 750Km   100个range门
N = Tw*fs;   %一个脉冲内的点数
K = Bw/Tw;   %调频斜率
t = (-N/2:N/2-1)'/fs;
Ndelay1 = fs*t_delay1;
Ndelay2 = fs*t_delay2;
fd1 = fd; %目标1doppler频率
%目标1 150Km，doppler10Hz
%目标2 750Km，doppler0Hz
amp = amp_min + (amp_max-amp_min)*rand();
%% 
C = 3e8;
RngRes = C/2/Bw/1e3; %range分辨率（Km）
DopRes = 1/Np/Tw;
rngAx = (0:N/2-1)*RngRes;
dopAx = (-Np/2:Np/2-1)*DopRes;
N = floor(N);
Np = floor(Np);
%% 本振
sig1Pulse = exp(1i*pi*(K*t.^2));        %本振
sigLo = repmat(sig1Pulse,Np,1);
% figure;plot(real(sigLo))
%% 构造仿真信号
%假目标
tar1 = sigLo;
Dop1 = exp(1i*2*pi*fd1*Tw*(0:Np-1));%exp(1j*2*pi*fd1*t);
Dop1 = repmat(Dop1,N,1);%repmat(Dop1,Np,1);
tar1 = circshift(tar1,Ndelay1);    %循环移位，相当于时延
% tar2=tar1;
tar1 = tar1.*Dop1(:);         % 添加doppler信息
tar1(1:Ndelay1) = 0;             %对应时延的部分置0，相当于噪声
%tar1 = awgn(tar1,10,'measured'); %加噪声
A_noise = 10; %噪声功率
tar1=amp*tar1+A_noise*randn(Np*N,1);%原来是0.1
data_target = tar1;
data_target = reshape(data_target,N,Np);
out_sig = data_target;
% figure;plot(real(tar1))
%真实目标

% tar2 = sigLo;
% tar2 = circshift(tar2,Ndelay1);    %循环移位，相当于时延
% tar2(1:Ndelay1) = 0;             %对应时延的部分置0，相当于噪声
% %tar1 = awgn(tar1,10,'measured'); %加噪声
% % tar2=tar2+0.1*randn(Np*N,1);
% tar2=tar2+A_noise*randn(Np*N,1);
% % A_noise = 1/sqrt(2); %噪声功率
% % tar2=tar2+A_noise*(randn(Np*N,1) + 1i*randn(Np*N,1));
% % data = tar2+amp*tar1;
% data = tar2;
% data = reshape(data,N,Np);
% data_add = tar2+tar1;
% data_add_1 = reshape(data_add,N,Np);

% figure;
% plot(real(data_target(:,1)),'r')%干扰
% hold on
% plot(real(data(:,1)),'b')
% xlabel('采样点数')
% ylabel('幅度')
% legend('目标','干扰')
% title('时域')
% 
% data_fft=fftshift(fft(data(:,1),2000));
% data_target_fft=fftshift(fft(data_target(:,1),2000));
% 
% figure;
% f1 = linspace(-fs/2,fs/2,2000);
% plot(f1,db(data_target_fft),'r');
% hold on
% plot(f1,db(data_fft),'b');
% xlabel('频率(Hz)');
% ylabel('幅度（dB）');
% legend('目标','干扰')
% title('目标和干扰累加频谱')
% 
% win_sz = 16; nfft =2048; nooverlap = win_sz - 1;
%     [S, F, T] = spectrogram(data_add_1(:,1), win_sz, nooverlap, nfft, fs,'centered');%
%     figure('Name','STFT时频谱'); imagesc(T, F, abs(S)); colorbar; set(gca, 'YDir', 'normal');
%     title('信号时频'); xlabel('时间/s'); ylabel('频率/Hz') ;
% %% range处理
% winRng = hanning(N);
% dataRng0 = zeros(N,Np);
% for iP = 1:Np
%     data_tmp = conj(data(:,iP)).*sig1Pulse;
%     dataRng0(:,iP) = fftshift(fft(data_tmp.*winRng));
% end
% dataRng = conj(dataRng0(N/2+1:end,:));  %这里取共轭将doppler调正
% % 
% % figure;
% % plot(rngAx,db(dataRng(:,1)))
% % xlabel('range/Km')
% % ylabel('功率/dB')
% % title('range压缩图')
% 
% % figure
% % mesh(db(dataRng))
% % shading interp
% % view(2)
% 
% %% doppler处理
% winDop = hanning(Np).';
% dataRD = fftshift(fft(dataRng.*repmat(winDop,N/2,1),[],2),2);
% out_sig=dataRD;
% 
% % figure
% % plot(dopAx,db(dataRD(21,:)))
% % xlabel('doppler/Hz')
% % ylabel('功率/dB')
% % title('目标1doppler切面')
% 
% % figure
% % plot(dopAx,db(dataRD(101,:)))
% % xlabel('doppler/Hz')
% % ylabel('功率/dB')
% % title('目标2doppler切面')
% %%%8/10
% % figure
% % surf(dopAx,rngAx,db(dataRD))
% % shading interp
% % view(2)
% % xlabel('doppler/Hz')
% % ylabel('range/Km')
% % xlim([-15,15])
% % ylim([1,1000])
% % title('RD图')
% 
% % figure
% % plot(rngAx,db(dataRD(:,513)))
% % xlabel('range/Km')
% % title('目标2range切面')
% data1=tar1;
% data1 = reshape(data1,N,Np);
% % figure;plot(real(data(:,1)))
% %% range处理
% winRng = hanning(N);
% dataRng0 = zeros(N,Np);
% for iP = 1:Np
%     data_tmp = conj(data1(:,iP)).*sig1Pulse;
%     dataRng0(:,iP) = fftshift(fft(data_tmp.*winRng));
% end
% dataRng = conj(dataRng0(N/2+1:end,:));  %这里取共轭将doppler调正
% % 
% % figure;
% % plot(rngAx,db(dataRng(:,1)))
% % xlabel('range/Km')
% % ylabel('功率/dB')
% % title('range压缩图')
% 
% % figure
% % mesh(db(dataRng))
% % shading interp
% % view(2)
% 
% %% doppler处理
% winDop = hanning(Np).';
% dataRD1 = fftshift(fft(dataRng.*repmat(winDop,N/2,1),[],2),2);
% out_sig=dataRD1;
% 
% % figure
% % plot(dopAx,db(dataRD1(21,:)))
% % xlabel('doppler/Hz')
% % ylabel('功率/dB')
% % title('干扰doppler切面')
% 
% figure; 
% plot(dopAx,db(dataRD(21,:)),'b')
% hold on
% plot(dopAx,db(dataRD1(21,:)),'r')
% xlabel('doppler/Hz')
% ylabel('功率/dB')
% legend('目标','干扰')
% title('速度拖doppler切面')

end

