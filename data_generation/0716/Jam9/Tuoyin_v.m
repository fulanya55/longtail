function out_sig = Tuoyin_v(npp,fs,Tp,K,sudu, juli,amp,f0)
% 单个速度假目标干扰 生成函数
% ----Input----
%   npp：脉冲重复周期点数
%   fs：标量。采样率。
%   Tp：脉宽
%   K： 调频斜率
%   sudu:假目标速度
%   juli：假目标距离
%   amp：假目标幅度
%   f0：载频
% ----Output----
%   out_sig： 输出数据。维数【nBatch×nSample】
Np = Tp * fs; 
tp = (-Np/2 : Np/2-1)' / fs;
fudu=amp;
lamuda=3e8/f0;
f0=2*sudu/lamuda;
sigLFM = fudu*exp(1i*pi*K*tp.^2+2*1i*pi*f0*tp);
delay = 2*juli/3e8;
n_delay = round(fs*delay);
pin_1=zeros(1,round(npp-n_delay));
pin_2=zeros(1,round(npp-length(sigLFM)-length(pin_1)));
% length(sigLFM)
% length(pin_1)
% length(pin_2)
zong=[pin_1,sigLFM.',pin_2];

out_sig=awgn((zong),20,'measured');



end