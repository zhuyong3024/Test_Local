%% train_NN_DPD_ILA.m
% 目的：基于当前 PA 模型，构建 ILA 训练数据，训练 NN 后失真补偿器

clear; clc;

%% 1. 产生训练用 OFDM 信号 + PA 输出
NR    = initNR_OFDM();
tx_bb = genOFDMFrame(NR);

y_pa  = pa_memory_poly(tx_bb);   % PA 输出（无 DPD）

% 为了提高训练稳定性，可以丢弃开头一小段过渡样本
Ntotal = length(tx_bb);
Nskip  = 200;   % 可调整
x_ref  = tx_bb(1+Nskip:end);
y_obs  = y_pa(1+Nskip:end);
N      = length(x_ref);

fprintf('有效训练样本数 N = %d\n', N);

%% 2. 构造训练特征（带记忆的 PA 输出）和目标（理想输入）

K = 3;   % 记忆深度（样点数），可调：3/5/7 等
numFeat = 2 * K;

% 为了构造 K 阶记忆，需要从第 K 个样点开始
idx_start = K;
M = N - K + 1;     % 可用样本数

X_feat = zeros(numFeat, M);  % 特征：numFeat × M
Y_tgt  = zeros(2,       M);  % 目标：2 × M（分别为 Re{x}, Im{x}）

for m = 1:M
    n = idx_start - 1 + m;   % 对应原始序列的时刻
    
    % 构造 [y[n], y[n-1], ..., y[n-K+1]]
    seg = y_obs(n:-1:n-K+1);      % K×1
    % 拼成 [Re(y[n]), Im(y[n]), Re(y[n-1]), Im(y[n-1]), ...]
    feat = zeros(numFeat,1);
    for k = 1:K
        feat(2*k-1) = real(seg(k));
        feat(2*k)   = imag(seg(k));
    end
    X_feat(:,m) = feat;
    
    % 目标是理想输入 x_ref(n)
    Y_tgt(1,m) = real(x_ref(n));
    Y_tgt(2,m) = imag(x_ref(n));
end

fprintf('构造特征矩阵 X_feat 大小 = [%d × %d]\n', size(X_feat,1), size(X_feat,2));
fprintf('构造目标矩阵 Y_tgt  大小 = [%d × %d]\n', size(Y_tgt,1),  size(Y_tgt,2));

%% 3. 转换为 cell 数组格式，供 trainNetwork 使用
% 
% M = size(X_feat,2);
% 
% XTrain = cell(1,M);
% YTrain = cell(1,M);
% 
% for m = 1:M
%     XTrain{m} = X_feat(:,m);   % [numFeat×1]
%     YTrain{m} = Y_tgt(:,m);    % [2×1]
% end
% 
% % 可选：打乱顺序以提高训练泛化
% idxPerm   = randperm(M);
% XTrain    = XTrain(idxPerm);
% YTrain    = YTrain(idxPerm);

%% 3. 转换为 trainNetwork 所需的数值矩阵格式
% 对于 featureInputLayer：
%   - XTrain: N × numFeatures
%   - YTrain: N × numResponses

M       = size(X_feat, 2);   % 样本数
numFeat = size(X_feat, 1);   % 特征维度

% 当前 X_feat 是 [numFeat × M]，转置成 [M × numFeat]
XTrain = X_feat.';   % [M × numFeat]
YTrain = Y_tgt.';    % [M × 2]

% 打乱样本顺序（按行打乱）
idxPerm = randperm(M);
XTrain  = XTrain(idxPerm, :);
YTrain  = YTrain(idxPerm, :);

fprintf('XTrain 大小 = [%d × %d]\n', size(XTrain,1), size(XTrain,2));
fprintf('YTrain 大小 = [%d × %d]\n', size(YTrain,1), size(YTrain,2));

%% 4. 定义网络结构（后失真补偿器：y_obs → x_ref）

numFeat = size(X_feat,1);

layers = [
    featureInputLayer(numFeat, 'Name','input')
    fullyConnectedLayer(64, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(64, 'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(2,  'Name','fc_out')   % 输出 2 维：Re, Im
    regressionLayer('Name','regout')];

%% 5. 训练参数设置
miniBatchSize = 1024;
maxEpochs     = 20;
lr            = 1e-3;

options = trainingOptions('adam', ...
    'InitialLearnRate', lr, ...
    'MaxEpochs',        maxEpochs, ...
    'MiniBatchSize',    miniBatchSize, ...
    'Shuffle',          'every-epoch', ...
    'Plots',            'training-progress', ...
    'Verbose',          true);

%% 6. 启动训练
netPost = trainNetwork(XTrain, YTrain, layers, options);

%% 7. 保存训练好的网络
if ~exist('../nn','dir')
    mkdir('../nn');
end
save('../nn/net_dpd_post.mat','netPost','K','NR');
