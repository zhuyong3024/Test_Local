%% analyze_signal_range.m
% 目的：分析 NN-DPD 链路中各主要信号的动态范围，为定点设计提供依据

clear; clc;

%% 1. 生成一帧新的 OFDM 信号
NR    = initNR_OFDM();
tx_bb = genOFDMFrame(NR);

% 去掉前 Nskip 点避免 PA 初始过渡
Nskip    = 200;
x_ref_all = tx_bb(1+Nskip:end);

%% 2. 无 DPD 的 PA 输出
y_no_all = pa_memory_poly(x_ref_all);

%% 3. 有 DPD 的 PA 输出（采用已经验证过的 NN-DPD 结构）
% 这里直接调用你之前的测试脚本逻辑
load('../nn/net_dpd_post.mat', 'netPost', 'K');

% 构造 NN-DPD 输入特征（与 test_NN_predistorter 一致）
K        = 3;               % 如果和 netPost 里的 K 不一致，请以 netPost 的为准
numFeat  = 2 * K;
N        = length(x_ref_all);
idx_start = K;
M         = N - K + 1;

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
x_dpd = YPred(:,1) + 1j*YPred(:,2);   % NN-DPD 输出，长度 M

% 经过 PA
y_dpd_all = pa_memory_poly(x_dpd);

%% 4. 对齐长度（方便统一统计）
x_ref = x_ref_all(K:end);   % 长度 M
y_no  = y_no_all(K:end);    % 长度 M
y_dpd = y_dpd_all;          % 长度 M

Nmin  = min([length(x_ref), length(y_no), length(y_dpd)]);
x_ref = x_ref(1:Nmin);
y_no  = y_no(1:Nmin);
y_dpd = y_dpd(1:Nmin);
x_dpd = x_dpd(1:Nmin);

%% 5. 统计每个信号的幅度范围
signals = {'x_ref','x_dpd','y_no','y_dpd'};
S       = struct();

for k = 1:numel(signals)
    name = signals{k};
    v    = eval(name);
    amp  = abs(v);
    S.(name).rms    = rms(v);
    S.(name).maxAbs = max(amp);
    S.(name).p999   = prctile(amp, 99.9);  % 99.9% 分位数
end

%% 6. 打印结果
fprintf('=== 信号动态范围统计 ===\n');
for k = 1:numel(signals)
    name = signals{k};
    st   = S.(name);
    fprintf('%s: RMS = %.4f, Max|x| = %.4f, 99.9%%|x| = %.4f\n', ...
        name, st.rms, st.maxAbs, st.p999);
end

%% 7. 可选：画幅度直方图
figure; 
for k = 1:numel(signals)
    name = signals{k};
    v    = eval(name);
    amp  = abs(v);
    subplot(2,2,k);
    histogram(amp, 100);
    title(sprintf('%s |x|', name));
    xlabel('Amplitude'); ylabel('Count');
end
sgtitle('NN-DPD 链路主要信号幅度直方图');