%% test_NN_predistorter.m
% 目的：把 netPost 当作预失真器使用，离线验证 NN-DPD 的效果
% 流程：
%   1) 生成新的 OFDM 信号 x_ref_all
%   2) 无 DPD：x_ref_all -> PA -> y_noDPD
%   3) 有 DPD：x_ref_all -> NN-DPD(netPost) -> PA -> y_dpd
%   4) 对齐长度后，比较 EVM / ACLR

clear; clc;

%% 1. 载入网络和参数
load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');

Fs = NR.Fs;
BW_main = 98e6;   % 主信道带宽，用之前的 ACLR 定义

fprintf('加载网络成功：记忆阶数 K = %d\n', K);

%% 2. 生成新的 OFDM 测试信号
tx_bb_test = genOFDMFrame(NR);   % 新一帧 OFDM

% 去掉前 Nskip 样点，避免 PA 初始过渡
Nskip     = 200;
x_ref_all = tx_bb_test(1+Nskip:end);
N         = length(x_ref_all);

fprintf('预失真测试：有效输入样本数 N = %d\n', N);

%% 3. 无 DPD 基准：直接经过 PA
y_no_all = pa_memory_poly(x_ref_all);  % 无 DPD 输出，长度 N

%% 4. 构造 NN-DPD 输入特征（以 x_ref_all 为"期望 PA 输出"）

numFeat   = 2 * K;
idx_start = K;
M         = N - K + 1;           % 由于需要 K 阶记忆，前 K-1 个样本不用

X_feat = zeros(numFeat, M);      % [2K × M]

for m = 1:M
    n = idx_start - 1 + m;       % 对应原序列时刻 n = K..N
    seg = x_ref_all(n:-1:n-K+1); % 取 K 个带记忆样本

    feat = zeros(numFeat,1);
    for k = 1:K
        feat(2*k-1) = real(seg(k));
        feat(2*k)   = imag(seg(k));
    end
    X_feat(:,m) = feat;
end

% 转成 [M × numFeat] 形式
X_dpd = X_feat.';    % 预失真器的输入矩阵

%% 5. 使用 netPost 作为 NN-DPD：x_ref -> x_dpd
% 注意：此处采用 ILA 思路，把学到的"PA 逆"直接用作预失真器

YPred = predict(netPost, X_dpd);           % [M × 2]
x_dpd = YPred(:,1) + 1j*YPred(:,2);       % NN-DPD 输出（PA 输入）

%% 6. NN-DPD + PA 链路
y_dpd_all = pa_memory_poly(x_dpd);        % 有 DPD 输出，长度约为 M

%% 7. 对齐长度，计算 EVM / ACLR

% 对齐到同一长度：
%  - x_ref：从第 K 个样本开始，与 X_dpd 对齐
%  - 无 DPD：丢掉前 K-1 个样本
%  - 有 DPD：y_dpd_all 长度为 M，与 x_ref(K:end) 对齐

x_ref = x_ref_all(K:end);          % 长度 M
y_no  = y_no_all(K:end);           % 长度 M
y_dpd = y_dpd_all;                 % 长度 M

% ------- 无 DPD：EVM / ACLR -------
g0   = (x_ref' * y_no) / (x_ref' * x_ref);
y0_l = g0 * x_ref;
err0 = y_no - y0_l;
EVM0 = sqrt(mean(abs(err0).^2) / mean(abs(y0_l).^2)) * 100;

ACLR0 = myACLR_fft(y_no, Fs, BW_main);

% ------- 有 DPD：EVM / ACLR -------
g1   = (x_ref' * y_dpd) / (x_ref' * x_ref);
y1_l = g1 * x_ref;
err1 = y_dpd - y1_l;
EVM1 = sqrt(mean(abs(err1).^2) / mean(abs(y1_l).^2)) * 100;

ACLR1 = myACLR_fft(y_dpd, Fs, BW_main);

%% 8. 打印结果
fprintf('\n=== NN 预失真离线测试结果 ===\n');
fprintf('无 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM0, ACLR0);
fprintf('有 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM1, ACLR1);
