% 参数设置
fs = 1e6; % 采样频率 1 MHz
T = 1e-3; % 信号持续时间 1 ms
t = 0:1/fs:T-1/fs; % 时间向量

% 生成线性调频信号（LFM）
fc = 100e3; % 载频 100 kHz
B = 500e3; % 带宽 500 kHz
S = B/T; % 扫频斜率
tx_signal = exp(1j*pi*S*t.^2) .* exp(1j*2*pi*fc*t); % 发射信号

% 假设目标距离和速度
R = 1000; % 目标距离 1000 m
v = 10; % 目标速度 10 m/s
tau = 2*R / 3e8; % 双程延迟
fd = 2*v / 3e8 * fc; % 多普勒频移

% 生成接收信号
rx_signal = circshift(tx_signal, [0, round(tau*fs)]) .* exp(1j*2*pi*fd*t);

% 生成RD谱
N = length(t);
range_fft = fft(rx_signal, N); % 距离维FFT
range_doppler_matrix = zeros(N, N);
for k = 1:N
    range_doppler_matrix(:, k) = fft(range_fft .* exp(1j*2*pi*(k-1)/N*t), N);
end

% 计算RD谱的幅度
rd_spectrum = abs(range_doppler_matrix);

% 绘制RD谱
figure;
imagesc(linspace(-fs/2, fs/2, N), linspace(0, T*fs, N), fftshift(rd_spectrum));
xlabel('Doppler Frequency (Hz)');
ylabel('Range (samples)');
title('Range-Doppler Spectrum');
colorbar;