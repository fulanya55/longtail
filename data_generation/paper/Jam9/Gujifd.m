clear; clc;
c = 3e8;                    % 光速 (m/s)
fc = 10e9;                   % 载频 (Hz)
B = 5e6;                     % 带宽 (Hz)
T = 100e-6;                  % 脉冲宽度 (s)
fs = 2 * B;                  % 采样率 (Hz)
t = -T/2 : 1/fs : T/2-1/fs;  % 时间轴

% 目标参数
v = 300;                     % 目标速度 (m/s)，靠近为正
R0 = 10e3;                   % 初始距离 (m)

% 生成LFM信号
k = B / T;                   % 调频斜率
s_tx = exp(1j * pi * k * t.^2) .* exp(1j * 2 * pi * fc * t);

tau = 2 * (R0 + v * t) / c;          % 时变时延
s_rx = interp1(t, s_tx, t - tau, 'linear', 0);  % 时延插值
s_rx = awgn(s_rx, 20, 'measured');   % 添加噪声

N = length(s_tx);
S_tx = fft(s_tx, 2*N);
S_rx = fft(s_rx, 2*N);
S_pc = S_rx .* conj(S_tx);
s_pc = ifft(S_pc);
s_pc = s_pc(1:N);   
% 峰值检测法
[~, idx] = max(abs(s_pc));
t_axis = (-N/2:N/2-1) * (1/fs);      % 时间轴
fd_estimated = k * t_axis(idx);      % 多普勒频率估计

% 理论值对比
fd_theory = 2 * v * fc / c;
fprintf('理论多普勒频率: %.2f Hz\n', fd_theory);
fprintf('估计多普勒频率: %.2f Hz\n', fd_estimated);

v_estimated = fd_estimated * c / (2 * fc);
fprintf('理论目标速度: %.2f m/s\n', v);
fprintf('估计目标速度: %.2f m/s\n', v_estimated);

% 时域信号对比
figure;
subplot(2,1,1);
plot(t, real(s_tx)), title('发射信号（实部）'), xlabel('时间 (s)');

% 脉冲压缩结果
subplot(2,1,2);
plot(t_axis, abs(s_pc)), hold on;
plot(t_axis(idx), abs(s_pc(idx)), 'ro');
title('脉冲压缩输出'), xlabel('时间 (s)'), legend('压缩结果', '峰值点');

% 频谱分析（验证多普勒）
figure;
[S_rx, f] = pwelch(s_rx, 256, [], [], fs);
plot(f - fs/2, fftshift(10*log10(S_rx)));
xlabel('频率 (Hz)'), ylabel('功率谱密度 (dB)');
title('回波信号频谱');