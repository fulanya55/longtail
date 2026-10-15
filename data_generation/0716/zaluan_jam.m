function [out_sig] = zaluan_jam(sigLFM,npp,fs,B,Tpp,n_min,n_max,A_min,A_max,T_min,T_max,tao_min,tao_max)
%SHUZHUANG_JAM 杂乱脉冲干扰
%   npp:点数
%   fs：采样率
%   B:带宽
% n:频点个数
%   Tpp:脉周期
% A：幅度
% T：杂乱脉冲时延
% tao:杂乱脉冲长度
Np = Tpp * fs; 
tp = (-Np/2 : Np/2-1)' / fs;
n_n=n_min + (n_max-n_min)*rand();
pd=rand(floor(n_n),1)*B;
he=zeros(1,floor(npp));
for i=1:n_n
    A=A_min + (A_max-A_min)*rand();
    T=T_min + (T_max-T_min)*rand();
    tao=tao_min + (tao_max-tao_min)*rand();
    T=floor(T*fs);
    tao=floor(tao*fs);
    pin1=1:T;
    mai=T+1:T+tao;
    if T+tao<npp
        chuang=[zeros(1,length(pin1)),ones(1,length(mai)),zeros(1,length(T+tao+1:npp))];
    else
        chuang=[zeros(1,length(pin1)),ones(1,floor(npp-length(pin1)))];
    end

    sig=A*exp(1i*2*pi*pd(i)*tp+rand()*2*pi*1i).*chuang.';
    he=he+sig.';
end
out_sig=he;
% out_sig=he.*sigLFM.';
out_sig=awgn(out_sig,20,'measured');
end

