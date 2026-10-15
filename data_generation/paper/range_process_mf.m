function data_rng = range_process_mf(LOSsig, sig2Proc, nRng2save)
% LFM脉冲串的距离处理函数，匹配滤波方法（与之对应的是去斜率法）。
% ----输入----
%   LOSig：        列向量。本振信号，长度为一个脉冲重复周期。
%   sig2Proc：     列向量。待处理数据。  调整时可用repmat函数，或foo(:)语法实现。
%   nRng2save：    正整数。要保留的距离门数。距离处理完后，远区距离可以不要，根据需求设定该数值。
% ----输出----
%   data_rng：     二维矩阵，脉冲数×距离点数。
% 2018.03.29 byTP
if ~isvector(LOSsig)
    error('ERROR: \n LOS signal must be a column vector!')
elseif ~isvector(sig2Proc)
    error('ERROR: \n Signal to process must be a column vector!')
else
    LOSsig = LOSsig(:);     %强制转换成列向量
    sig2Proc = sig2Proc(:); %强制转换成列向量
end

nPP = length(sig2Proc);
nFFT = nPP;

tmp = ifft(conj(fft(LOSsig, nFFT)) .* fft(sig2Proc, nFFT));    %频域方法实现线性卷积.这种更好！本振信号带不带0都能得到正确结果。下面的方法带0结果正确，不带0结果错误。
% tmp = ifft((fft(conj(flipud(LOSsig)), nFFT)) .* fft(sig2Proc, nFFT)); %频域方法实现线性卷积,这两种方法都对
data_rng = tmp(1 : 1+nRng2save-1);
data_rng = data_rng.';
end
 