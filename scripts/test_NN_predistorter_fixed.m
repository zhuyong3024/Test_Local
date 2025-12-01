%% test_NN_predistorter_fixed.m
% 目的：评估不同定点位宽对 NN-DPD 性能的影响
%      对比：
%        1) 浮点无 DPD / 有 DPD
%        2) B = 10/12/14/16 位定点下，无 DPD / 有 DPD

clear; clc;

%% 1. 加载网络与系统参数
load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');

Fs      = NR.Fs;
BW_main = 98e6;
Nskip   = 200;   % 去掉前 200 个过渡样本

fprintf('使用的记忆阶数 K = %d\n', K);

%% 2. 生成一帧新的 OFDM 信号
tx_bb = genOFDMFrame(NR);

% 去掉前 Nskip 个样本
x_ref_all = tx_bb(1+Nskip:end);
N_all     = length(x_ref_all);

%% 3. 浮点无 DPD 基准
y_no_all = pa_memory_poly(x_ref_all);

%% 4. 浮点有 DPD（与 test_NN_predistorter 基本相同）

% 构造 NN-DPD 输入特征
numFeat   = 2 * K;
idx_start = K;
M         = N_all - K + 1;

X_feat = zeros(numFeat, M);
for m = 1:M
    n = idx_start - 1 + m;
    seg = x_ref_all(n:-1:n-K+1);
    feat = zeros(numFeat,1);
    for k = 1:K
        feat(2*k-1) = real(seg(k));
        feat(2*k)   = imag(seg(k));
    end
    X_feat(:,m) = feat;
end
X_dpd = X_feat.';   % [M × numFeat]

YPred = predict(netPost, X_dpd);
x_dpd = YPred(:,1) + 1j*YPred(:,2);
y_dpd_all = pa_memory_poly(x_dpd);

% 对齐长度（从第 K 个样本开始）
x_ref = x_ref_all(K:end);    % 长度 M
y_no  = y_no_all(K:end);     % 长度 M
y_dpd = y_dpd_all;           % 长度 M

Nmin  = min([length(x_ref), length(y_no), length(y_dpd)]);
x_ref = x_ref(1:Nmin);
y_no  = y_no(1:Nmin);
y_dpd = y_dpd(1:Nmin);

% 浮点无 DPD
g0   = (x_ref' * y_no) / (x_ref' * x_ref);
y0_l = g0 * x_ref;
err0 = y_no - y0_l;
EVM0 = sqrt(mean(abs(err0).^2)/mean(abs(y0_l).^2)) * 100;
ACLR0 = myACLR_fft(y_no, Fs, BW_main);

% 浮点有 DPD
g1   = (x_ref' * y_dpd) / (x_ref' * x_ref);
y1_l = g1 * x_ref;
err1 = y_dpd - y1_l;
EVM1 = sqrt(mean(abs(err1).^2)/mean(abs(y1_l).^2)) * 100;
ACLR1 = myACLR_fft(y_dpd, Fs, BW_main);

fprintf('\n=== 浮点基准 ===\n');
fprintf('浮点 无 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM0, ACLR0);
fprintf('浮点 有 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM1, ACLR1);

%% 5. 定点仿真参数
B_list     = [10 12 14 16];   % 尝试的位宽
xmax_in    = 2.0;             % DPD 输入全幅
xmax_dpd   = 1.2;             % DPD 输出全幅

fprintf('\n=== 定点仿真结果 ===\n');

for B = B_list
    % 5.1 量化 DPD 输入（x_ref_all）
    x_ref_all_q = quant_uniform(x_ref_all, B, xmax_in);

    % 5.2 用量化后的 x_ref_all_q 计算无 DPD 输出
    y_no_all_q = pa_memory_poly(x_ref_all_q);

    % 5.3 用量化后的 x_ref_all_q 作为 NN-DPD 输入
    %     注意：特征也用量化信号构造
    X_feat_q = zeros(numFeat, M);
    for m = 1:M
        n = idx_start - 1 + m;
        seg_q = x_ref_all_q(n:-1:n-K+1);
        feat_q = zeros(numFeat,1);
        for k = 1:K
            feat_q(2*k-1) = real(seg_q(k));
            feat_q(2*k)   = imag(seg_q(k));
        end
        X_feat_q(:,m) = feat_q;
    end
    X_dpd_q = X_feat_q.';    % [M × numFeat]

    % 5.4 NN-DPD 输出（仍为浮点，但输入是量化的）
    YPred_q = predict(netPost, X_dpd_q);
    x_dpd_q = YPred_q(:,1) + 1j*YPred_q(:,2);

    % 5.5 对 DPD 输出也做量化，模拟 FPGA/DAC 精度
    x_dpd_q2 = quant_uniform(x_dpd_q, B, xmax_dpd);

    % 5.6 通过 PA 得到有 DPD 输出（定点链路）
    y_dpd_all_q = pa_memory_poly(x_dpd_q2);

    % 5.7 对齐长度，从第 K 个样本开始
    x_ref_q = x_ref_all_q(K:end);
    y_no_q  = y_no_all_q(K:end);
    y_dpd_q = y_dpd_all_q;

    Nmin_q  = min([length(x_ref_q), length(y_no_q), length(y_dpd_q)]);
    x_ref_q = x_ref_q(1:Nmin_q);
    y_no_q  = y_no_q(1:Nmin_q);
    y_dpd_q = y_dpd_q(1:Nmin_q);

    % 5.8 计算"量化无 DPD"的 EVM/ACLR
    g0_q   = (x_ref_q' * y_no_q) / (x_ref_q' * x_ref_q);
    y0_l_q = g0_q * x_ref_q;
    err0_q = y_no_q - y0_l_q;
    EVM0_q = sqrt(mean(abs(err0_q).^2)/mean(abs(y0_l_q).^2)) * 100;
    ACLR0_q = myACLR_fft(y_no_q, Fs, BW_main);

    % 5.9 计算"量化有 DPD"的 EVM/ACLR
    g1_q   = (x_ref_q' * y_dpd_q) / (x_ref_q' * x_ref_q);
    y1_l_q = g1_q * x_ref_q;
    err1_q = y_dpd_q - y1_l_q;
    EVM1_q = sqrt(mean(abs(err1_q).^2)/mean(abs(y1_l_q).^2)) * 100;
    ACLR1_q = myACLR_fft(y_dpd_q, Fs, BW_main);

    % 5.10 打印结果
    fprintf('B = %2d 位：无 DPD EVM = %7.2f %%, ACLR = %6.2f dB | 有 DPD EVM = %7.2f %%, ACLR = %6.2f dB\n', ...
        B, EVM0_q, ACLR0_q, EVM1_q, ACLR1_q);
end