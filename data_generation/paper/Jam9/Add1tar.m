function out_sig = Add1tar(input_sig, sig_LOS, nPulse, fs)
%% 给仿真出来的纯干扰添加目标信号，使得结果更加接近真实情况
% 高斯白噪声功率为1。
% 按照要求：目标和噪声同量级;干扰和目标相对距离0-3公里；角度主要考虑主瓣内。
% ----Input----
%   input_sig：输入的干扰信号
%   sig_LOS：本振信号
%   nPulse:一个样本里的脉冲重复周期个数。
%   fs:采样率
% ----Output----
%   out_sig：添加了目标的干扰信号
%   20180914 by TP
global SNR_addTar; %信噪比
%% 强制转换成行向量
input_sig = input_sig(:).';
sig_LOS = sig_LOS(:).';
%% 参数设置：距离差、方位差（方位暂时没考虑）
C = 3e8;
A_tar = 10^(SNR_addTar/20);        %目标信号幅度
tar_rng = 3000*rand();      %目标距离，0~3公里内随机
tar_delay = tar_rng/C;      %转换成时间
tar_delay_index = round(fs*tar_delay); %转换成采样点
%% 目标信号构造
tar_sig = A_tar * circshift(repmat(sig_LOS, 1, 1), tar_delay_index);
%% 目标和干扰累加
out_sig = input_sig + tar_sig;
end