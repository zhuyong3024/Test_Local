function tx_bb = genOFDMFrame(NR)
% genOFDMFrame  生成 5G-like OFDM 基带时域信号
%
% 使用方法：
%   NR    = initNR_OFDM;
%   tx_bb = genOFDMFrame(NR);
%
% 输出：
%   tx_bb : 带循环前缀的复杂基带时域信号（列向量）

%% 1. 基本参数
Nfft = NR.Nfft;
Nsc  = NR.Nsc;                       % 占用子载波数（含 DC）
Nsym = NR.NsymPerSlot * NR.Nslot;    % 总 OFDM 符号数
M    = NR.ModOrder;                  % QAM 阶数

%% 2. 生成 QAM 符号（仅放在数据子载波上）
bitsPerSym = log2(M);
numBits    = NR.Ndata * bitsPerSym * Nsym;

% 随机比特流（列向量）
dataBits = randi([0 1], numBits, 1, 'uint8');

% 使用 qammod 做比特输入的 QAM 调制
dataSymAll = qammod( ...
    dataBits, M, ...
    'InputType', 'bit', ...
    'UnitAveragePower', true);

% 每列是一个 OFDM 符号
dataSym = reshape(dataSymAll, NR.Ndata, Nsym);

%% 3. 映射到频域子载波：加保护带 + DC 空载
Xk = complex(zeros(Nfft, Nsym));     % 频域网格

guardL  = NR.GuardBands(1);
guardH  = NR.GuardBands(2);
usedIdx = (guardL+1):(Nfft-guardH);  % 共 Nsc 个 index

if NR.DCnull
    % 中间一个 subcarrier 空出来做 DC
    Ndata = NR.Ndata;                % = Nsc - 1
    Nleft  = floor(Ndata/2);
    Nright = Ndata - Nleft;

    % usedIdx 结构：
    % [ 左 Nleft 数据 ][ DC ][ 右 Nright 数据 ]
    dcPos    = usedIdx(Nleft+1);     % DC 位置
    leftIdx  = usedIdx(1:Nleft);
    rightIdx = usedIdx(Nleft+2:end); % 跳过 DC

    data_left  = dataSym(1:Nleft,   :);
    data_right = dataSym(Nleft+1:end, :);

    Xk(leftIdx,  :) = data_left;
    Xk(rightIdx, :) = data_right;
    Xk(dcPos,    :) = 0;             % DC 空载
else
    Xk(usedIdx, :) = dataSym;
end

%% 4. IFFT 到时域 + 加循环前缀
% ifftshift 把 DC 移到正确位置
xt = ifft(ifftshift(Xk, 1), Nfft, 1);   % Nfft × Nsym

Ncp   = NR.CP_len;
xt_cp = [xt(end-Ncp+1:end, :); xt];     % (Nfft+Ncp) × Nsym

% 串行化为列向量
tx_bb = xt_cp(:);

%% 5. 简单归一化（控制平均功率）
tx_bb = tx_bb / rms(tx_bb) * sqrt(0.5); % 平均功率约 0.5

end