% signal = data(:,1:3000)+1j*data(:,3001:6000);
% %% a 
% C = 3e8;
% Tp = 100e-6;     %脉宽
% Tpp = 300e-6;    %脉冲重复周期
% duty_cycle = Tp/Tpp; %占空比
% % Tpp = 100e-6;    %脉冲重复周期
% fs = 10e6;      %采样率
% Bw = 5e6;      %带宽
% Np = Tp * fs;     %脉冲内点数
% nPP = ceil(Tpp * fs);   %脉冲重复周期内点数
% K = Bw / Tp;      %调频斜率
% % tp = (-Np/2 : Np/2-1)' / fs;
% tp = (1 : Np)' / fs;
% % t = (-nPP/2: nPP/2-1)/fs;
% t = (1: nPP)/fs;
% R = t*C/2;
% R_max = Tpp*C/2;
% R_det = C/2/Bw;
% %% 本振信号
% sigLFM = exp(1i*pi*K*tp.^2);  %线性调频信号
% SigLFM = abs(fftshift(fft(sigLFM)));
% sig_LOS = [sigLFM ; zeros(ceil(nPP-Np), 1)];  %单脉冲信号
% 选择干扰类型
n = 7;
n1 = (n-1)*400+80;

a = signal(n1,:);
figure()
plot(real(a(1,:)));
a_f = fft(a,[],2);
figure()
plot(db(a_f(1,:)));
for i = 1:1
    a_r(i,:) = range_process_mf(a(i,:).',sig_LOS.',nPP);
end
figure()
plot(db(ifftshift(a_r(1,:))));
figure()
% [S, f, t] = spectrogram(a(1,:), hamming(128), 120, nPP, fs,'center');
[S, f, t] = spectrogram(a(1,:), hamming(4), 2, nPP, fs,'center');
imagesc(db(S));