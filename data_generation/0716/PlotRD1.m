clear; clc;
c = 3e8;                     % 光速 (m/s)
fc = 10e9;                   % 载频 (Hz)
B = 5e6;                     % 带宽 (Hz)
T_pulse = 50e-6;             % 脉冲宽度 (s)
PRF = 2000;                  % 脉冲重复频率 (Hz)
fs = 2 * B;                  % 采样率 (Hz)
N_pulse = 64;                % 脉冲数
R0 = 10e3;                   % 目标初始距离 (m)
v = 150;                     % 目标速度 (m/s)，靠近为正

% 生成LFM脉冲
t_chirp = -T_pulse/2 : 1/fs : T_pulse/2-1/fs;
k = B / T_pulse;             % 调频斜率
s_tx = exp(1j * pi * k * t_chirp.^2) .* exp(1j * 2 * pi * fc * t_chirp);
% 生成脉冲序列
t_axis = (0 : N_pulse-1) / PRF;          % 慢时间轴 (脉冲间时间)
range_history = R0 - v * t_axis;          % 目标距离变化

s_rx = zeros(length(t_chirp), N_pulse);   % 接收信号矩阵（快时间×慢时间）
for n = 1:N_pulse
    tau = 2 * range_history(n) / c;       % 当前脉冲时延
    t_delayed = t_chirp - tau;
    s_rx(:, n) = exp(1j * pi * k * t_delayed.^2) .* exp(1j * 2 * pi * fc * t_delayed);
    s_rx(:, n) = awgn(s_rx(:, n), 20, 'measured'); % 添加噪声
end
% 匹配滤波器
h_mf = conj(fliplr(s_tx));                % 时域匹配滤波器
s_pc = zeros(size(s_rx));
for n = 1:N_pulse
    s_pc(:, n) = conv(s_rx(:, n), h_mf, 'same'); % 逐脉冲压缩
end
% 距离维FFT（可选，提高分辨率）
N_range = 2^nextpow2(size(s_pc, 1));
range_fft = fft(s_pc, N_range, 1);

% 多普勒维FFT
N_doppler = 2^nextpow2(size(s_pc, 2));
rd_spectrum = fftshift(fft(range_fft, N_doppler, 2), 2);

% 计算坐标轴
range_bin = (0:N_range-1) * (c / (2 * B));          % 距离轴 (m)
doppler_bin = (-N_doppler/2 : N_doppler/2-1) * (PRF / N_doppler); % 多普勒轴 (Hz)
velocity_bin = doppler_bin * (c / (2 * fc));        % 速度轴 (m/s)
figure;
imagesc(velocity_bin, range_bin, 20*log10(abs(rd_spectrum)));
xlabel('速度 (m/s)'), ylabel('距离 (m)'), title('RD谱');
colorbar, colormap('jet'), axis xy;

% 标记目标位置
[~, idx_range] = min(abs(range_bin - R0));
[~, idx_doppler] = min(abs(doppler_bin - (2*v*fc/c)));
hold on;
plot(velocity_bin(idx_doppler), range_bin(idx_range), 'rx', 'MarkerSize', 10);
legend('目标位置');
figure;
mesh(20*log10(abs(rd_spectrum)))