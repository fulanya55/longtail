function jam_sig = ISRJ(original_sig, tau, fs, nReplica)
%间歇采样重复转发干扰(JSRJ)生成函数
% ----Input----
% original_sig: 向量，待干扰的信号
% tau: 干扰采样脉冲宽度
% fs: 采样率
% nReplica: 干扰采样段的重复次数
% ----Output----
% jam_fig: 干扰信号

if nargin == 3 
    nReplica = 1;
end

if ~isvector(original_sig)
    error('ERROR: \n Original signal must be a column vector!')
else
    original_sig = original_sig(:);  %强制转换为列向量
end

tau_p = ceil(fs * tau);         %转发时长对应采样点数
nSig = length(original_sig); 
nJamSegment = ceil(nSig/(1+nReplica)/tau_p);   %干扰截取的段数
tmp_win = [ones(tau_p, 1) ; zeros(tau_p*nReplica, 1)];    %一段干扰的窗
jam_win = repmat(tmp_win, nJamSegment ,1);  %所有干扰的窗
jam_win = jam_win(1:nSig);     %只保留一个周期长度

jam_sample = jam_win .* original_sig; %干扰采样

jam_sig = zeros(nSig, 1);
for ii = 1:nReplica
    tmp_data = [zeros(ii*tau_p, 1) ; jam_sample(1:end-ii*tau_p)] ;
    jam_sig = jam_sig + tmp_data;   %干扰信号
end

end
