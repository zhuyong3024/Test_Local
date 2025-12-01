%% test_NN_postdistorter.m
% 目的：在"新的 OFDM 帧"上测试 netPost 的补偿效果
% 流程：生成新 tx_bb_test → 通过 PA 得到 y_pa_test →
%       用 netPost 估计 x_hat_test → 和真实 x_ref_test 比较 EVM/NMSE

clear; clc;

%% 1. 载入训练好的网络和参数
load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');

fprintf('加载网络成功：记忆阶数 K = %d\n', K);

%% 2. 生成一帧"全新"的 OFDM + PA 输出（与训练时不同的随机比特）
tx_bb_test = genOFDMFrame(NR);        % 新一帧 OFDM
y_pa_test  = pa_memory_poly(tx_bb_test);

% 丢弃前 Nskip 个样本，避免 PA 初始状态过渡影响
Nskip      = 200;
x_ref_all  = tx_bb_test(1+Nskip:end);
y_obs_all  = y_pa_test(1+Nskip:end);
N          = length(x_ref_all);

fprintf('测试样本总数（去掉前 %d 点）N = %d\n', Nskip, N);

%% 3. 构造测试特征：以 PA 输出 y_obs_all 为输入，带记忆 K
numFeat   = 2 * K;
idx_start = K;
M         = N - K + 1;         % 可用样本数

X_feat = zeros(numFeat, M);    % [2K × M]
x_ref  = zeros(M,1);           % 对齐后的参考 x(n)

for m = 1:M
    n = idx_start - 1 + m;     % 对应原序列时刻

    seg = y_obs_all(n:-1:n-K+1);  % K 个带记忆样本

    feat = zeros(numFeat,1);
    for k = 1:K
        feat(2*k-1) = real(seg(k));
        feat(2*k)   = imag(seg(k));
    end
    X_feat(:,m) = feat;

    x_ref(m) = x_ref_all(n);      % 目标：当前时刻的理想输入
end

% 转成 trainNetwork / predict 所需的 [M × numFeat] 数值矩阵
XTest = X_feat.';    % [M × numFeat]

%% 4. 用 netPost 做预测：y_obs → x_hat
YPred = predict(netPost, XTest);    % [M × 2]

x_hat = YPred(:,1) + 1j*YPred(:,2); % 还原成复数序列

%% 5. 计算 NMSE 与 EVM

% 方式一：NMSE（dB）
num = sum(abs(x_ref - x_hat).^2);
den = sum(abs(x_ref).^2);
NMSE_dB = 10*log10(num/den);

% 方式二：波形 EVM-like（允许一个最佳线性增益）
g = (x_ref' * x_hat) / (x_ref' * x_ref);   % 最佳线性缩放
x_hat_lin = g * x_ref;
err = x_hat - x_hat_lin;

EVM_wave = sqrt(mean(abs(err).^2)/mean(abs(x_hat_lin).^2)) * 100;

%% 6. 输出结果
fprintf('测试集 NMSE = %.2f dB\n', NMSE_dB);
fprintf('测试集 波形 EVM-like = %.2f %%\n', EVM_wave);
