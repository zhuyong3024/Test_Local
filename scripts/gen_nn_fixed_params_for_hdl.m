function gen_nn_fixed_params_for_hdl(B_W, B_A)
% gen_nn_fixed_params_for_hdl  生成用于 HDL/Simulink 的定点 NN 参数
%
% 输入：
%   B_W : 权重量化位宽（与 test_NN_predistorter_fixedNN 中选定的一致）
%   B_A : 激活量化位宽（可以先传 10，保持一致）
%
% 输出：
%   ../hdl/nn_dpd_fixed_params.mat 文件，内部为 fi 类型的 W1/b1/W2/b2/W3/b3
%   （位宽足够大，不再强行压成 Q1.11）

%% 1. 加载训练好的网络
S = load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');
netPost = S.netPost;

layers = netPost.Layers;
fc1    = layers(2);   % 'fc1'
fc2    = layers(4);   % 'fc2'
fc3    = layers(6);   % 'fc_out'

W1 = fc1.Weights;  % [64 × numFeat]
b1 = fc1.Bias(:);  % [64 × 1]
W2 = fc2.Weights;  % [64 × 64]
b2 = fc2.Bias(:);  % [64 × 1]
W3 = fc3.Weights;  % [2 × 64]
b3 = fc3.Bias(:);  % [2 × 1]

%% 2. 采用与 nn_dpd_forward_fixed.m 一致的 xmax 配置
xmax_W1 = 0.4;
xmax_b1 = 0.6;
xmax_W2 = 0.2;
xmax_b2 = 0.8;
xmax_W3 = 0.4;
xmax_b3 = 0.1;

% 激活的 xmax（虽然这里不保存，但保持一致以便以后需要）
xmax_A1 = 0.8;
xmax_A2 = 0.7;

%% 3. 使用 quant_uniform_real 对权重/偏置做量化（double 结果）
addpath('../scripts');  % 确保能找到 quant_uniform_real.m

W1_q = quant_uniform_real(W1, B_W, xmax_W1);
b1_q = quant_uniform_real(b1, B_W, xmax_b1);
W2_q = quant_uniform_real(W2, B_W, xmax_W2);
b2_q = quant_uniform_real(b2, B_W, xmax_b2);
W3_q = quant_uniform_real(W3, B_W, xmax_W3);
b3_q = quant_uniform_real(b3, B_W, xmax_b3);

%% 4. 把这些量化后的 double 转成高精度 fi（比如 sfix23_En20）
WL = 23;
FL = 20;
T  = numerictype(1, WL, FL);
F  = fimath( ...
    'RoundingMethod', 'Nearest', ...
    'OverflowAction', 'Saturate', ...
    'ProductMode', 'FullPrecision', ...
    'SumMode', 'FullPrecision');

W1_fi = fi(W1_q, T, F);
b1_fi = fi(b1_q, T, F);
W2_fi = fi(W2_q, T, F);
b2_fi = fi(b2_q, T, F);
W3_fi = fi(W3_q, T, F);
b3_fi = fi(b3_q, T, F);

%% 5. 保存到 hdl 目录
if ~exist('../hdl','dir')
    mkdir('../hdl');
end

save('../hdl/nn_dpd_fixed_params.mat', ...
    'W1_fi','b1_fi','W2_fi','b2_fi','W3_fi','b3_fi');

fprintf('已生成 ../hdl/nn_dpd_fixed_params.mat，B_W = %d, B_A = %d\n', B_W, B_A);
end
