function NR = initNR_OFDM()
% initNR_OFDM  初始化 100 MHz 5G-like OFDM 参数（R2025b）
%
% 使用方法：
%   NR = initNR_OFDM;

% 基本参数
NR.fc   = 3.5e9;       % 载频（这里只在文档层面用，基带仿真实际不用）
NR.SCS  = 30e3;        % 子载波间隔 30 kHz
NR.Fs   = 122.88e6;    % 采样率（100 MHz 带宽常用的 Fs）
NR.Nfft = 4096;        % IFFT 大小

% 资源块等
NR.NRB  = 273;                 % 100 MHz，对应 273 RB
NR.Nsc  = 12 * NR.NRB;         % 占用子载波数 = 3276
NR.DCnull = true;              % 是否空掉 DC 子载波
NR.Ndata  = NR.Nsc - NR.DCnull; % 实际承载数据的子载波数量

% Guard band 设计：满足 sum(GuardBands)+Nsc = Nfft
remain = NR.Nfft - NR.Nsc;     % 剩余子载波数
assert(remain > 0 && mod(remain,2)==0, 'Nfft 与 Nsc 不匹配');
NR.GuardBands = [remain/2, remain/2];  % 例如 [410 410]

% OFDM 符号相关
NR.NsymPerSlot = 14;           % 每个 slot 14 个符号
NR.CP_ratio    = 1/14;         % 简化：CP 比例，后面可按标准细化
NR.CP_len      = round(NR.Nfft * NR.CP_ratio);

% 调制与帧结构（先简单设置，将来可按需要扩展）
NR.ModOrder = 16;              % 16-QAM
NR.Nslot    = 5;               % 模拟 5 个 slot（可自行调节）

% 方便使用的小量
NR.Ts = 1/NR.Fs;

end