%% analyze_net_activations.m
% 目的：在一帧 OFDM 上分析 netPost 各层激活的动态范围，为激活量化设计提供依据

clear; clc;

%% 1. 加载网络与系统参数
S = load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');
netPost = S.netPost;
K       = S.K;
NR      = S.NR;

fprintf('网络记忆阶数 K = %d\n', K);

%% 2. 生成一帧 OFDM，并构造特征 X_dpd（与 DPD 训练/测试一致）

tx_bb = genOFDMFrame(NR);

% 去掉前 Nskip 点
Nskip     = 200;
x_ref_all = tx_bb(1+Nskip:end);
N_all     = length(x_ref_all);

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

fprintf('特征矩阵 X_dpd 尺寸 = [%d × %d]\n', size(X_dpd,1), size(X_dpd,2));

%% 3. 从 netPost 中取出各全连接层权重/偏置

layers = netPost.Layers;

fc1 = layers(2);   % 'fc1'
fc2 = layers(4);   % 'fc2'
fc3 = layers(6);   % 'fc_out'

W1 = fc1.Weights;  % [64 × numFeat]
b1 = fc1.Bias;     % [64 × 1]

W2 = fc2.Weights;  % [64 × 64]
b2 = fc2.Bias;     % [64 × 1]

W3 = fc3.Weights;  % [2 × 64]
b3 = fc3.Bias;     % [2 × 1]

%% 4. 手工前向传播，统计激活范围

% fc1: Z1 = X_dpd * W1.' + b1.',  A1 = relu(Z1)
Z1 = X_dpd * W1.' + repmat(b1.', size(X_dpd,1), 1);   % [M × 64]
A1 = max(Z1, 0);   % ReLU

% fc2: Z2 = A1 * W2.' + b2.',  A2 = relu(Z2)
Z2 = A1 * W2.' + repmat(b2.', size(A1,1), 1);         % [M × 64]
A2 = max(Z2, 0);   % ReLU

% fc_out: Z3 = A2 * W3.' + b3.', 输出层不需要统计
Z3 = A2 * W3.' + repmat(b3.', size(A2,1), 1);         % [M × 2]

%% 5. 统计 A1 / A2 的幅度（这里只看实数，直接 abs 即可）

sigNames = {'A1','A2'};
for k = 1:numel(sigNames)
    name = sigNames{k};
    V    = eval(name);   % [M × 64]
    v    = V(:);         % 展成一维

    amp  = abs(v);
    rms_v    = rms(v);
    max_v    = max(amp);
    p999_v   = prctile(amp, 99.9);

    fprintf('\n[%s] 激活范围统计:\n', name);
    fprintf('  RMS = %.4f,  Max = %.4f,  99.9%% = %.4f\n', rms_v, max_v, p999_v);
end

%% 6. 可选：画直方图
figure;
subplot(2,1,1);
histogram(A1(:), 100);
title('A1 (fc1 ReLU 输出)'); xlabel('Amplitude'); ylabel('Count');

subplot(2,1,2);
histogram(A2(:), 100);
title('A2 (fc2 ReLU 输出)'); xlabel('Amplitude'); ylabel('Count');
