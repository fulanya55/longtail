function out_sig = Traction_duration(sig_LOS,Jam_LOS, nPulse, fs)
% 拖引时长参数提取函数
% ----Input----
%   sig_LOS：行向量矩阵。拖引干扰信号，一个周期。
%   nPulse:一个拖引周期内脉冲个数。
%   fs：标量。采样率。
% ----Output----
%   out_sig： 输出数据。输入拖引时间
Jam_LOS = Jam_LOS;
nPP = length(Jam_LOS(1,:));%一个脉冲的点数
tpp = nPP/fs;%脉冲重复频率
%%频谱峰值检测多普勒频移
N = length(Jam_LOS(1,:));
for i = 1:nPulse
    S_dx(i,:) = range_process_mf(Jam_LOS(i,:).',sig_LOS.',nPP);
%     figure();
%     plot(db(S_dx(i,:)));
end


% for i = 1:nPulse
%     S_tx = fft(Jam_LOS(i,:));
%     [~, idx] = max(abs(S_tx(1:nPP/2)));
%     t_axis = (-N/2:N/2-1) * (1/fs);      % 时间轴
%     fd_estimated(i) = idx;
% end
%%希尔伯特变换估计瞬时频率
% for i = 1:nPulse
%     analytic_signal = hilbert(Jam_LOS(i,:));  % 计算解析信号
%     phase = unwrap(angle(analytic_signal));  % 计算瞬时相位
%     instantaneous_freq = diff(phase)/(2*pi)*fs;  % 计算瞬时频率
%     figure()
%     plot(instantaneous_freq)
% end
%%短时傅里叶变化
for i = 1:nPulse
% for i = 14:14
[S, f, t] = spectrogram(Jam_LOS(i,:), hamming(256), 250, nPP, fs,'center');
for ii = 1:size(S,1)
    D(ii) = sum(abs(S(ii,:)));
end
% figure()
% plot(db(D));
% figure()
% imagesc(db(S));
[~, idx] = find(db(D)>(max(db(D))-8));  % 找到每个时间点最大能量对应的频率索引
% figure()
% plot(max(db(S)));
instantaneous_freq = round(idx);
% instantaneous_freq = f(idx);  % 转换为频率值
% figure();
% plot(instantaneous_freq);
fd_estimated(i) = mean(instantaneous_freq);
end


    % S1 = fftshift(fft(S_dx,nPulse,1));
    % figure;
    % mesh(db(S1));

%%估计距离移动
for i = 1:nPulse
    [~, idx] = max(abs(S_dx(i,:)));
    rp_estimated(i) = abs(idx); % 估计的多普勒频移
end
%比较多普勒与距离变化确定拖引时长
for i = 1:nPulse-1
    fd(i) = fd_estimated(i+1) - fd_estimated(i);
    rp(i) = rp_estimated(i+1) - rp_estimated(i);
end
fd_t = length(fd(fd~=0));
rp_t = length(rp(rp~=0));
out_sig = [fd_t*tpp,rp_t*tpp];
% out_sig = rp_t*tpp;
