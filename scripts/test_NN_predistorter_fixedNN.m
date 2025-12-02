%% test_NN_predistorter_fixedNN.m
% 目的：比较 浮点 NN-DPD 与 定点 NN-DPD 在系统 EVM/ACLR 上的差异
%% 1. 加载网络和系统参数
S = load('../nn/net_dpd_post.mat', 'netPost', 'K', 'NR');
netPost = S.netPost;
K       = S.K;
NR      = S.NR;

Fs      = NR.Fs;
BW_main = 98e6;
Nskip   = 200;

fprintf('使用的记忆阶数 K = %d\n', K);

%% 2. 生成一帧新的 OFDM
tx_bb = genOFDMFrame(NR);

% 去掉前 Nskip 样本
x_ref_all = tx_bb(1+Nskip:end);
N_all     = length(x_ref_all);

%% 3. 无 DPD 输出（浮点 PA）
y_no_all = pa_memory_poly(x_ref_all);

%% 4. 构造特征矩阵 X_dpd（与之前一致）
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

%% 5. 浮点 NN-DPD 作为参考
YPred_float = predict(netPost, X_dpd);
x_dpd_float = YPred_float(:,1) + 1j*YPred_float(:,2);
y_dpd_all_float = pa_memory_poly(x_dpd_float);

% 对齐长度
x_ref = x_ref_all(K:end);    % 长度 M
y_no  = y_no_all(K:end);     % 长度 M
y_dpd_float = y_dpd_all_float;

Nmin  = min([length(x_ref), length(y_no), length(y_dpd_float)]);
x_ref = x_ref(1:Nmin);
y_no  = y_no(1:Nmin);
y_dpd_float = y_dpd_float(1:Nmin);

% 无 DPD
g0   = (x_ref' * y_no) / (x_ref' * x_ref);
y0_l = g0 * x_ref;
err0 = y_no - y0_l;
EVM0 = sqrt(mean(abs(err0).^2)/mean(abs(y0_l).^2)) * 100;
ACLR0 = myACLR_fft(y_no, Fs, BW_main);

% 浮点 有 DPD
g1f   = (x_ref' * y_dpd_float) / (x_ref' * x_ref);
y1_lf = g1f * x_ref;
err1f = y_dpd_float - y1_lf;
EVM1_float = sqrt(mean(abs(err1f).^2)/mean(abs(y1_lf).^2)) * 100;
ACLR1_float = myACLR_fft(y_dpd_float, Fs, BW_main);

fprintf('\n=== 浮点基准 ===\n');
fprintf('浮点 无 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM0,        ACLR0);
fprintf('浮点 有 DPD：EVM = %.2f %%, ACLR = %.2f dB\n', EVM1_float, ACLR1_float);

%% 6. 不同权重量化位宽下的 定点 NN-DPD

B_W_list = [6 8 10 12];   % 权重量化位宽
B_A      = 10;            % 激活位宽，固定为 10 位

fprintf('\n=== 定点 NN-DPD 结果（仅量化权重+激活） ===\n');

for B_W = B_W_list
    % 6.1 定点 NN 前向推理
    x_dpd_fixed = nn_dpd_forward_fixed(X_dpd, B_W, B_A);

    % 6.2 通过同一个 PA 得到输出
    y_dpd_all_fixed = pa_memory_poly(x_dpd_fixed);

    % 6.3 对齐长度
    y_dpd_fixed = y_dpd_all_fixed(1:Nmin);

    % 6.4 计算有 DPD（定点 NN）的 EVM/ACLR
    g1   = (x_ref' * y_dpd_fixed) / (x_ref' * x_ref);
    y1_l = g1 * x_ref;
    err1 = y_dpd_fixed - y1_l;
    EVM1 = sqrt(mean(abs(err1).^2)/mean(abs(y1_l).^2)) * 100;
    ACLR1 = myACLR_fft(y_dpd_fixed, Fs, BW_main);

    fprintf('B_W = %2d 位：定点 NN 有 DPD：EVM = %7.2f %%, ACLR = %6.2f dB\n', ...
        B_W, EVM1, ACLR1);
end
