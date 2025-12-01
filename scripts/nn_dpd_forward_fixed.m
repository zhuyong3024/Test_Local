function x_dpd = nn_dpd_forward_fixed(X_dpd, B_W, B_A)
% nn_dpd_forward_fixed  定点 NN-DPD 前向推理（离线仿真用）
%
% 输入：
%   X_dpd : [M × numFeat] 特征矩阵（每行对应 1 个样本）
%   B_W   : 权重量化位宽（含符号位），例如 6, 8, 10, 12
%   B_A   : 激活量化位宽，例如 10, 12
%
% 输出：
%   x_dpd : [M × 1] 复数输出，作为 PA 的输入（定点 NN 的结果）

%% 1. 加载网络
S = load('../nn/net_dpd_post.mat', 'netPost');
netPost = S.netPost;

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

[M, numFeat] = size(X_dpd); %#ok<NASGU>

%% 2. 各层权重/偏置量化范围（根据 analyze_net_weights 的统计）

% fc1
xmax_W1 = 0.4;
xmax_b1 = 0.6;

% fc2
xmax_W2 = 0.2;
xmax_b2 = 0.8;

% fc_out
xmax_W3 = 0.4;
xmax_b3 = 0.1;

% 激活
xmax_A1 = 0.8;
xmax_A2 = 0.7;

%% 3. 量化权重和偏置
W1_q = quant_uniform_real(W1, B_W, xmax_W1);
b1_q = quant_uniform_real(b1, B_W, xmax_b1);

W2_q = quant_uniform_real(W2, B_W, xmax_W2);
b2_q = quant_uniform_real(b2, B_W, xmax_b2);

W3_q = quant_uniform_real(W3, B_W, xmax_W3);
b3_q = quant_uniform_real(b3, B_W, xmax_b3);

%% 4. 前向传播 + 激活量化

% fc1: Z1 = X_dpd * W1_q.' + b1_q.', A1 = relu(Z1), 再量化
Z1 = X_dpd * W1_q.' + repmat(b1_q.', size(X_dpd,1), 1);   % [M × 64]
A1 = max(Z1, 0);                                          % ReLU
A1_q = quant_uniform_real(A1, B_A, xmax_A1);              % 激活量化

% fc2: Z2 = A1_q * W2_q.' + b2_q.', A2 = relu(Z2), 再量化
Z2 = A1_q * W2_q.' + repmat(b2_q.', size(A1_q,1), 1);     % [M × 64]
A2 = max(Z2, 0);                                          % ReLU
A2_q = quant_uniform_real(A2, B_A, xmax_A2);              % 激活量化

% fc_out: Z3 = A2_q * W3_q.' + b3_q.'
Z3 = A2_q * W3_q.' + repmat(b3_q.', size(A2_q,1), 1);     % [M × 2]

% 最终输出仍用浮点表示，但已包含"定点 NN"的误差
x_dpd = Z3(:,1) + 1j*Z3(:,2);

end
