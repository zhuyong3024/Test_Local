function acpr_dB = myACLR_fft(x, Fs, BW_main)
% myACLR_fft  基于 FFT 的 ACLR 计算（带主信道带宽参数）
%
%   acpr_dB = myACLR_fft(x, Fs, BW_main)
%
% 定义：
%   - 主信道：|f| <= BW_main/2
%   - 邻道： BW_main/2 < |f| <= Fs/2 （两侧合并为一个"等效单侧邻道"）
%
% 注意：
%   - BW_main 必须小于 Fs（否则主信道会占满整个 Nyquist 区间）

x = x(:);                  % 列向量
Nfft = 2^nextpow2(length(x));

Xf = fftshift(fft(x, Nfft));
P  = abs(Xf).^2;           % 功率谱（未归一化）

% 频率轴（Hz）
f = (-Nfft/2:Nfft/2-1)'/Nfft * Fs;

if BW_main >= Fs
    error('BW_main 必须小于采样率 Fs');
end

halfMain = BW_main / 2;

% 主信道：|f| <= BW_main/2
idx_main = abs(f) <= halfMain;

% 邻道：BW_main/2 < |f| <= Fs/2
idx_adj  = (abs(f) > halfMain) & (abs(f) <= Fs/2);

P_main = sum(P(idx_main));
P_adj_total = sum(P(idx_adj));

% 两侧频带合成一个"等效单侧邻道"
P_adj = P_adj_total / 2;

acpr_dB = 10 * log10(P_adj / P_main);

end