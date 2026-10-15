function out_sig = sudujmb(npp,fs,Tp,K,delay_min, delay_max, amp_min, amp_max,v_min,v_max,n_fake_tar_min ,n_fake_tar_max )
% 单个速度假目标干扰 生成函数
% ----Input----
%   npp：点数
%   fs：标量。采样率。
%   Tp：脉宽
%   K： 调频斜率
%   delay_min：标量，单位s，假目标时延最小值
%   delay_max：标量，单位s，假目标时延最大值
%   amp_min：标量，假目标幅度最小值
%   amp_max：标量，假目标幅度最大值
%   v_min：速度最小值
%   v_max：速度最大值
% ----Output----
%   out_sig： 输出数据。维数【nBatch×nSample】
Np = Tp * fs; 
tp = (-Np/2 : Np/2-1)' / fs;
shumu=n_fake_tar_min + (n_fake_tar_max-n_fake_tar_min)*rand();
for ii=1:shumu
fudu=amp_min + (amp_max-amp_min)*rand();
f0=v_min + (v_max-v_min)*rand();
sigLFM = fudu*exp(1i*pi*K*tp.^2+2*1i*pi*f0*tp); 
delay = delay_min + (delay_max-delay_min)*rand();
n_delay = round(fs*delay);
pin_1=zeros(1,round(npp-n_delay));
pin_2=zeros(1,round(npp-length(sigLFM)-length(pin_1)));
% length(sigLFM)
% length(pin_1)
% length(pin_2)
zong(ii,:)=[pin_1,sigLFM.',pin_2];
end

out_sig=awgn(sum(zong),20,'measured');

end

