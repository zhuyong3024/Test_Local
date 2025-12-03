%% export_nn_fixed_for_hdl.m（废弃废弃废弃）
% 目的：从 net_dpd_post.mat 中导出 NN-DPD 权重/偏置，
%       按 nn_dpd_forward_fixed.m 中的定点设计进行量化，
%       并转成 fi，供 Simulink + HDL Coder 使用。

clear; clc;

%% 1. 加载网络
S = load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');
netPost = S.netPost;
K       = S.K;

layers = netPost.Layers;
fc1 = layers(2);   % 对照 nn_dpd_forward_fixed.m
fc2 = layers(4);
fc3 = layers(6);   % fc_out

W1 = fc1.Weights;  % [64 × numFeat]
b1 = fc1.Bias;     % [64 × 1]
W2 = fc2.Weights;  % [64 × 64]
b2 = fc2.Bias;     % [64 × 1]
W3 = fc3.Weights;  % [2 × 64]
b3 = fc3.Bias;     % [2 × 1]

%% 2. 选定定点位宽（和你在 test_NN_predistorter_fixedNN 中选的一致）
B_W = 12;   % <<== 根据你的实验结果修改
B_A = 12;   % <<== 激活位宽，后面定义中间层 numerictype 用

% 可以把 nn_dpd_forward_fixed.m 中的 xmax_* 复制过来，保证行为一致
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

%% 3. 使用已有的均匀量化函数量化为 double
addpath('../scripts');  % 确保能找到 quant_uniform_real.m

W1_q = quant_uniform_real(W1, B_W, xmax_W1);
b1_q = quant_uniform_real(b1, B_W, xmax_b1);

W2_q = quant_uniform_real(W2, B_W, xmax_W2);
b2_q = quant_uniform_real(b2, B_W, xmax_b2);

W3_q = quant_uniform_real(W3, B_W, xmax_W3);
b3_q = quant_uniform_real(b3, B_W, xmax_b3);

%% 4. 转成 fi（方便 HDL Coder 知道是定点）
T_w = numerictype(1, B_W, B_W-1);  % Q1.(B_W-1)，适合 |W| < 1 的情况
F_w = fimath('RoundingMethod','Nearest', ...
             'OverflowAction','Saturate', ...
             'ProductMode','FullPrecision', ...
             'SumMode','FullPrecision');

W1_fi = fi(W1_q, T_w, F_w);
b1_fi = fi(b1_q, T_w, F_w);
W2_fi = fi(W2_q, T_w, F_w);
b2_fi = fi(b2_q, T_w, F_w);
W3_fi = fi(W3_q, T_w, F_w);
b3_fi = fi(b3_q, T_w, F_w);

%% 5. 保存到 hdl 目录，供 Simulink 使用
if ~exist('../hdl','dir')
    mkdir('../hdl');
end

save('../hdl/nn_dpd_fixed_params.mat', ...
     'W1_fi','b1_fi','W2_fi','b2_fi','W3_fi','b3_fi', ...
     'B_W','B_A','K');
